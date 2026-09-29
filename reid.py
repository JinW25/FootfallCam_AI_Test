"""
reid.py
-------
Turns a cropped person image into a fixed-length "appearance embedding" and
compares embeddings for similarity. 

This now uses a real pretrained deep network (ImageNet ResNet18, run
through ONNX Runtime) as a fixed feature extractor. 
Run `python setup_reid_model.py` once before using
this module -- it downloads and prepares
models/resnet18_embedding.onnx.

"""

import json
import os

import cv2
import numpy as np
import onnxruntime as ort

# REID_MODEL_PATH lets you point at a fine-tuned model (see train_reid_head.py)
# without editing this file -- e.g. `set REID_MODEL_PATH=models\finetuned_reid.onnx`
# (Windows) or `export REID_MODEL_PATH=models/finetuned_reid.onnx` (macOS/Linux)
# before running main.py. Unset, this uses the original generic model.
_MODEL_PATH = os.environ.get(
    "REID_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models",
                 "resnet18_embedding.onnx"),
)
# the generic model's internal tensor name
_EMBEDDING_TENSOR_NAME = "onnx::Flatten_189"
# (see setup_reid_model.py). A fine-tuned model from train_reid_head.py exports with a
# single, already-clean output instead of this internal name -- _get_output_name() below
# detects which situation it's in automatically, so this file doesn't need to change
# depending on which model is currently loaded.
_INPUT_NAME = "input.0"
_INPUT_SIZE = 224
# Default (ResNet18/ImageNet-style) normalization
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_session = None  # lazy singleton -- loaded once, on first use
_output_name = None


def _load_normalization_stats():
    global _MEAN, _STD
    meta_path = os.path.splitext(_MODEL_PATH)[0] + ".meta.json"
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        _MEAN = np.array(meta["mean"], dtype=np.float32)
        _STD = np.array(meta["std"], dtype=np.float32)
        print(f"reid.py: loaded normalization stats from {meta_path} "
              f"(mean={meta['mean']}, std={meta['std']})")


def _get_session():
    global _session, _output_name
    if _session is None:
        _load_normalization_stats()
        if not os.path.exists(_MODEL_PATH):
            raise RuntimeError(
                f"{_MODEL_PATH} not found. Run `python3 setup_reid_model.py` "
                f"once to download and prepare it (or set REID_MODEL_PATH to "
                f"point at a fine-tuned model from train_reid_head.py)."
            )
        _session = ort.InferenceSession(
            _MODEL_PATH, providers=["CPUExecutionProvider"])
        output_names = [o.name for o in _session.get_outputs()]
        if _EMBEDDING_TENSOR_NAME in output_names:
            _output_name = _EMBEDDING_TENSOR_NAME  # the original generic model
        elif len(output_names) == 1:
            # a fine-tuned model -- single clean output
            _output_name = output_names[0]
        else:
            raise RuntimeError(
                f"Can't tell which of this model's outputs is the embedding: "
                f"{output_names}. Set _EMBEDDING_TENSOR_NAME or export the model "
                f"with a single output."
            )
    return _session


def embed(crop):
    """
    Build an embedding from a BGR person crop by running it through
    ResNet18 and taking the 512-dim feature vector from just before the
    final classification layer (see setup_reid_model.py for why that
    layer, not the 1000-way class output, is what's used) -- or, if
    REID_MODEL_PATH points at a fine-tuned model, that model's own output
    directly.
    """
    if crop is None or crop.size == 0:
        return None

    # BGR to RGB colour conversion, OpenCV loads images as BGR, the network expects RGB
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Resize to 224 x 224 (the fixed input size the CNN backbone expects)
    img = cv2.resize(img, (_INPUT_SIZE, _INPUT_SIZE))
    # Normalizing per channel, using default ImageNet statistics for the ResNet18
    img = (img - _MEAN) / _STD
    # Transpose from HWC to CHW layout
    img = img.transpose(2, 0, 1)[None].astype(np.float32)  # HWC -> NCHW

    session = _get_session()
    output = session.run([_output_name], {_INPUT_NAME: img})[0]
    return output.flatten().astype(np.float32)


def similarity(embedding_a, embedding_b):
    """
    Cosine similarity. Unlike the old color-histogram embedding, this is
    reported as the RAW cosine value rather than rescaled to [0, 1] --
    the pooled post-ReLU features this embedding uses are already
    non-negative in practice, so cosine similarity naturally lands in
    [0, 1] without needing an artificial rescale. (If you swap in a
    different embedding that isn't non-negative, revisit this.)
    similarity(a, b) = (a · b) / (‖a‖ × ‖b‖)
    """
    if embedding_a is None or embedding_b is None:
        return 0.0
    a, b = embedding_a, embedding_b
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def best_match(embedding, gallery):
    """
    Compare one embedding against a gallery (list) of reference embeddings
    and return (best_similarity, best_index). If the gallery is a single
    AVERAGED embedding (see load_reference_gallery(..., average=True)),
    this just compares against that one point. If it's several separate
    reference embeddings (average=False), this takes the max across them --
    a strong match to *any* known appearance of the person counts, rather
    than requiring a blended match to all of them at once.
    """
    if not gallery:
        return 0.0, -1
    sims = [similarity(embedding, ref) for ref in gallery]
    best_i = int(np.argmax(sims))
    return sims[best_i], best_i


def load_reference_gallery(ref_dir, average=True):
    """
    Loads every reference image in ref_dir, embeds each one, and returns
    the gallery best_match() expects.

    average=True (default): every reference embedding is L2-normalized
    (scaled to the same length) and then averaged into ONE combined
    embedding, returned as a single-item list. Normalizing first matters --
    without it, whichever reference photo happens to produce larger raw
    numbers (which can vary crop to crop for reasons that have nothing to
    do with identity -- lighting, crop size) would silently dominate the
    average more than the others. This treats every reference photo as
    evidence about the SAME person's one underlying appearance, which is
    the right choice when you have several photos of them and want the
    gallery to represent "this person," not "these specific outfits."

    Tested against this project's real calibration data (README.md's
    "Key finding #2" crops): averaging the two original references scored
    BETTER than keeping them separate in that specific test (true match
    0.787 vs. 0.741; fewer other-people crops scored above it: 12/37 vs.
    17/37) -- somewhat contrary to this project's own earlier assumption.
    A plausible reason: the two references shared a strong "general
    person" signal (build, overall tone) plus a divergent "jacket on/off"
    signal, and averaging reinforced the shared part while the divergent
    part partly canceled out.

    That result isn't guaranteed to hold for every set of references,
    though -- it depended on the two photos already being reasonably
    similar overall. Averaging references that capture MORE drastically
    different states of the person (a different colored shirt entirely, a
    very different angle) risks blending into a composite that doesn't
    strongly resemble any single real appearance, worse than any one
    photo alone would be. If your reference set includes photos that
    different from each other, average=False (keep every embedding
    separate, take the best match to any of them) is the safer choice --
    it costs nothing extra since best_match() already handles a gallery of
    any size.
    """
    embeddings = []
    for fname in sorted(os.listdir(ref_dir)):
        img = cv2.imread(os.path.join(ref_dir, fname))
        if img is not None:
            embeddings.append(embed(img))
    if not embeddings:
        raise RuntimeError(f"No reference images found in {ref_dir}")

    if not average:
        return embeddings

    normalized = [e / (np.linalg.norm(e) + 1e-8) for e in embeddings]
    avg = np.mean(normalized, axis=0)
    return [avg]
