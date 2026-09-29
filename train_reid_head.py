"""
train_reid_head.py
---------------------
Fine-tunes a small embedding head on top of the EXISTING frozen ResNet18
backbone, using triplet loss over the crops labeled with label_crops.py.
Designed specifically for a small dataset (tens to low hundreds of
labeled crops from one video), not hundreds of thousands of images:

  - The ResNet18 backbone itself is entirely FROZEN (no gradients flow
    into it at all). Only a small new head -- Linear(512,256) ->
    ReLU -> Linear(256,128), a few hundred thousand parameters -- gets
    trained. This is deliberately the most overfitting-resistant option:
    the general-purpose visual features (edges, textures, shapes) stay
    exactly as pretrained; only how those features get RE-COMBINED to
    separate this specific person from everyone else is learned.

  - Triplet loss, not a binary classifier. With N labeled crops, a
    classifier gets N training signals; triplet loss (anchor, positive,
    negative) gets up to N^3 -- much more signal per labeled image, which
    matters a lot when N is small. The loss is defined directly on COSINE
    similarity (not Euclidean distance) specifically because that's what
    reid.py's similarity()/best_match() use at inference time -- training
    and inference need to optimize the same thing.

  - Heavy data augmentation (random crop, flip, color jitter, small
    rotation) on every training image, specifically to squeeze more
    effective diversity out of a small dataset -- each labeled crop is
    seen in a different randomized variation every epoch, rather than the
    network just memorizing a handful of exact images.

  - A held-out validation split (never trained on) and early stopping,
    to actually catch overfitting rather than hope it doesn't happen --
    training stops when validation triplet accuracy stops improving, not
    after a fixed number of epochs regardless of whether it's still
    learning the person or starting to just memorize training crops.

Requires: pip install torch torchvision pillow

Usage:
    python train_reid_head.py --data_dir training_data --epochs 50
Produces:
    training_data/reid_head.pt          (PyTorch weights, for further training)
    models/finetuned_reid.onnx          (drop-in replacement -- see below)

To use the fine-tuned model:
    # Windows:
    set REID_MODEL_PATH=models\\finetuned_reid.onnx
    # macOS/Linux:
    export REID_MODEL_PATH=models/finetuned_reid.onnx
    python main.py --video sample.mp4 ...
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

INPUT_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FrozenBackbone(nn.Module):
    """The exact same ResNet18 features reid.py's generic model uses --
    frozen, not fine-tuned. Only the head below is trained."""

    def __init__(self):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(
            *list(base.children())[:-1])  # drop the final FC
        for p in self.features.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            return self.features(x).flatten(1)  # (batch, 512)


class ProjectionHead(nn.Module):
    """The only part that actually gets trained."""

    def __init__(self, in_dim=512, hidden=256, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        z = self.net(x)
        # unit length -- makes cosine similarity = dot product
        return F.normalize(z, dim=1)


def load_manifest_splits(data_dir, val_fraction=0.2, seed=0):
    manifest = json.load(
        open(os.path.join(data_dir, "candidates_manifest.json")))
    staff = [m for m in manifest if m["label"] == "staff"]
    not_staff = [m for m in manifest if m["label"] == "not_staff"]
    if len(staff) < 4 or len(not_staff) < 4:
        raise RuntimeError(
            f"Only {len(staff)} staff and {len(not_staff)} not_staff labeled crops found -- "
            f"need at least a handful of each (ideally 20+/50+) to train anything meaningful. "
            f"Run label_crops.py label to label more."
        )
    rng = random.Random(seed)
    rng.shuffle(staff)
    rng.shuffle(not_staff)
    n_val_staff = max(2, int(len(staff) * val_fraction))
    n_val_not = max(2, int(len(not_staff) * val_fraction))
    return {
        "train_staff": staff[n_val_staff:],
        "val_staff": staff[:n_val_staff],
        "train_not_staff": not_staff[n_val_not:],
        "val_not_staff": not_staff[:n_val_not],
    }


def make_transform(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_image(data_dir, entry, transform):
    img = Image.open(os.path.join(data_dir, entry["crop_path"])).convert("RGB")
    return transform(img)


def sample_triplet_batch(data_dir, staff_entries, not_staff_entries, transform, batch_size,
                         backbone, head):
    """Each triplet: anchor and positive are two DIFFERENT staff crops
    (forces the model to generalize across pose/lighting, not just match a
    crop to itself). The negative is chosen by HARD-NEGATIVE MINING, not
    pure random sampling -- see the module-level note on why this matters.
    """
    anchors, positives, negatives = [], [], []
    candidate_pool_size = min(len(not_staff_entries), 16)
    for _ in range(batch_size):
        a, p = random.sample(staff_entries, 2)
        a_img = load_image(data_dir, a, transform)
        p_img = load_image(data_dir, p, transform)

        # Hard-negative mining: instead of one random not_staff crop, pull a
        # POOL of candidates and keep whichever one the model currently
        # scores as MOST similar to the anchor -- the "hardest" (most
        # confusing) negative available this step. A random negative is
        # usually already easy to tell apart, so the model gets weak
        # gradient signal and can satisfy the loss cheaply by compressing
        # everything toward one region of space (collapse) rather than
        # learning real separation. Forcing it to repeatedly confront its
        # OWN current mistakes is what prevents that shortcut.
        pool = random.sample(not_staff_entries, candidate_pool_size)
        pool_imgs = torch.stack(
            [load_image(data_dir, m, transform) for m in pool])
        with torch.no_grad():
            a_emb = head(backbone(a_img.unsqueeze(0)))
            pool_embs = head(backbone(pool_imgs))
            sims = (a_emb * pool_embs).sum(dim=1)
            hardest_idx = int(sims.argmax())
        n_img = pool_imgs[hardest_idx]

        anchors.append(a_img)
        positives.append(p_img)
        negatives.append(n_img)
    return torch.stack(anchors), torch.stack(positives), torch.stack(negatives)


def cosine_triplet_loss(anchor, positive, negative, margin=0.4):
    """Defined on cosine similarity directly (embeddings are already unit
    length from ProjectionHead, so dot product = cosine similarity), to
    match what reid.py's similarity()/best_match() use at inference.

    margin raised from an earlier 0.2 -- too small a margin can be
    satisfied even by a collapsed embedding space (everything mapped
    close to the same point), since the loss only needs a tiny relative
    lean in the right direction once similarities are all compressed near
    1.0. A larger margin makes that cheap shortcut cost more loss, pushing
    training toward genuinely separating embeddings instead.
    """
    sim_pos = (anchor * positive).sum(dim=1)
    sim_neg = (anchor * negative).sum(dim=1)
    return F.relu(margin - sim_pos + sim_neg).mean()


def evaluate(backbone, head, data_dir, splits, transform, n_triplets=100):
    """Validation 'accuracy': fraction of held-out triplets where the
    trained embedding correctly scores the positive higher than the
    negative. Simple, directly interpretable, and matches what actually
    matters for this task."""
    head.eval()
    correct = 0
    with torch.no_grad():
        for _ in range(n_triplets):
            a, p = random.sample(splits["val_staff"], 2)
            n = random.choice(splits["val_not_staff"])
            imgs = torch.stack([
                load_image(data_dir, a, transform),
                load_image(data_dir, p, transform),
                load_image(data_dir, n, transform),
            ])
            emb = head(backbone(imgs))
            sim_pos = (emb[0] * emb[1]).sum().item()
            sim_neg = (emb[0] * emb[2]).sum().item()
            if sim_pos > sim_neg:
                correct += 1
    head.train()
    return correct / n_triplets


def export_onnx(backbone, head, out_path):
    class Combined(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            return self.head(self.backbone(x))

    combined = Combined(backbone, head).eval()
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    torch.onnx.export(
        combined, dummy, out_path,
        input_names=["input.0"], output_names=["embedding"],
        dynamic_axes=None, opset_version=18,
        # (was 13 -- too old for this PyTorch/onnxscript combo to write directly,
        # which forced an automatic downgrade-conversion step that then crashed
        # on a specific operator. 18 is what the exporter already wants to
        # produce natively, so this skips that broken conversion step entirely.
        # onnxruntime -- what reid.py actually uses at inference -- supports
        # opset 18 fine.)
        dynamo=False,  # recent PyTorch defaults to the newer "dynamo" export path,
        # which pulls in a separate onnxscript package this project doesn't otherwise
        # need at all. This is a plain feedforward CNN with no unusual control flow,
        # so the older, stable TorchScript-based exporter (dynamo=False) handles it
        # fine without that extra dependency.
    )
    print(f"Exported {out_path}")


def main(data_dir, epochs, batch_size, lr, patience):
    print("Loading frozen ResNet18 backbone (downloads pretrained weights on first run)...")
    backbone = FrozenBackbone()
    head = ProjectionHead()

    splits = load_manifest_splits(data_dir)
    print(
        f"Train: {len(splits['train_staff'])} staff, {len(splits['train_not_staff'])} not_staff")
    print(
        f"Val:   {len(splits['val_staff'])} staff, {len(splits['val_not_staff'])} not_staff")

    train_transform = make_transform(train=True)
    val_transform = make_transform(train=False)

    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    steps_per_epoch = max(4, len(splits["train_staff"]) * 2)

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        for _ in range(steps_per_epoch):
            a_imgs, p_imgs, n_imgs = sample_triplet_batch(
                data_dir, splits["train_staff"], splits["train_not_staff"],
                train_transform, batch_size, backbone, head,
            )
            a_emb = head(backbone(a_imgs))
            p_emb = head(backbone(p_imgs))
            n_emb = head(backbone(n_imgs))
            loss = cosine_triplet_loss(a_emb, p_emb, n_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc = evaluate(backbone, head, data_dir, splits, val_transform)
        print(f"epoch {epoch+1}/{epochs}: train_loss={total_loss/steps_per_epoch:.4f}, "
              f"val_triplet_accuracy={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(head.state_dict(), os.path.join(
                data_dir, "reid_head.pt"))
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"No improvement for {patience} epochs, stopping early "
                      f"(best val_triplet_accuracy={best_val_acc:.3f}).")
                break

    print(f"\nBest validation triplet accuracy: {best_val_acc:.3f}")
    print("(0.5 = no better than chance, 1.0 = perfect separation on held-out data)")
    if best_val_acc < 0.75:
        print("This is fairly low -- with this little data, consider labeling more crops "
              "(especially more DIFFERENT other people for not_staff) before trusting this "
              "model over the original generic one.")

    head.load_state_dict(torch.load(os.path.join(data_dir, "reid_head.pt")))

    check_embedding_collapse(backbone, head, data_dir, splits, val_transform)

    os.makedirs("models", exist_ok=True)
    export_onnx(backbone, head, os.path.join("models", "finetuned_reid.onnx"))


def check_embedding_collapse(backbone, head, data_dir, splits, transform, n_samples=15):
    """
    Validation triplet accuracy only ever compares ONE positive against ONE
    negative at a time -- it can look fine even when the embedding space
    has collapsed (everything mapped close to the same point), since
    relative ordering can survive even in a nearly-degenerate space. This
    catches that directly: compute pairwise similarity among several
    DIFFERENT not_staff crops -- people who are NOT each other and should
    score low against one another. If they instead all score close to 1.0,
    the model isn't discriminating people at all anymore, whatever the
    validation number said. Runs in seconds, so it's worth checking before
    committing to a multi-hour full-video run with a broken model.
    """
    print("\nChecking for embedding collapse (comparing several DIFFERENT "
          "not_staff crops against each other -- these should score LOW, "
          "not all bunched near 1.0)...")
    pool = random.sample(splits["val_not_staff"] + splits["train_not_staff"],
                         min(n_samples, len(splits["val_not_staff"]) + len(splits["train_not_staff"])))
    with torch.no_grad():
        imgs = torch.stack([load_image(data_dir, m, transform) for m in pool])
        embs = head(backbone(imgs))
        sim_matrix = embs @ embs.T
        n = sim_matrix.shape[0]
        off_diagonal = sim_matrix[~torch.eye(n, dtype=torch.bool)]
        mean_sim = off_diagonal.mean().item()
        min_sim = off_diagonal.min().item()
        max_sim = off_diagonal.max().item()

    print(
        f"Different-people similarity: mean={mean_sim:.3f}, range={min_sim:.3f}-{max_sim:.3f}")
    if mean_sim > 0.9:
        print("WARNING: different people are scoring extremely similar to each other "
              "(mean > 0.9). This strongly suggests embedding collapse -- the model has "
              "learned to map most crops close to the same point rather than actually "
              "separating people. Ranking many candidate tracks with this model will "
              "likely produce near-identical top scores, not a meaningful margin. Consider "
              "retraining -- hard-negative mining and a larger margin (both already in this "
              "version of the script) should help; if it still collapses, try labeling more "
              "varied not_staff crops, or reducing the head's capacity further.")
    else:
        print("Looks healthy -- different people are NOT scoring uniformly high.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=8,
                        help="stop early after this many epochs with no validation improvement")
    parser.add_argument("--export_only", action="store_true",
                        help="skip training entirely -- just load the already-saved "
                        "reid_head.pt (from a previous run) and (re-)export it to ONNX. "
                        "Useful if training succeeded but the export step failed "
                        "(e.g. a missing onnxscript dependency) -- no need to retrain.")
    parser.add_argument("--check_collapse_only", action="store_true",
                        help="skip training and export entirely -- just load the already-saved "
                        "reid_head.pt and run the embedding-collapse diagnostic against it. "
                        "Fast (seconds), useful for checking whether an already-trained "
                        "model is usable before running it on the full video.")
    args = parser.parse_args()

    if args.check_collapse_only:
        print("Loading frozen ResNet18 backbone...")
        backbone = FrozenBackbone()
        head = ProjectionHead()
        head.load_state_dict(torch.load(
            os.path.join(args.data_dir, "reid_head.pt")))
        splits = load_manifest_splits(args.data_dir)
        check_embedding_collapse(
            backbone, head, args.data_dir, splits, make_transform(train=False))
    elif args.export_only:
        print("Loading frozen ResNet18 backbone...")
        backbone = FrozenBackbone()
        head = ProjectionHead()
        head_path = os.path.join(args.data_dir, "reid_head.pt")
        head.load_state_dict(torch.load(head_path))
        print(f"Loaded trained head from {head_path}")
        os.makedirs("models", exist_ok=True)
        export_onnx(backbone, head, os.path.join(
            "models", "finetuned_reid.onnx"))
    else:
        main(args.data_dir, args.epochs, args.batch_size, args.lr, args.patience)
