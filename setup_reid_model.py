"""
setup_reid_model.py
--------------------
Downloads a pretrained ImageNet ResNet18 in ONNX format and performs a
small graph surgery step so its 512-dim pre-classification feature vector
(the output of the global-average-pool layer, right before the final
1000-way classifier) is exposed as a usable model output.

Why ResNet18 specifically, and why this over the classical descriptor:
--------------------------------------------------------------------------
reid.py originally used a hand-built HSV color histogram as its
"embedding". Calibrating it against this project's actual reference
photos showed it barely separates the true staff member from other
people in this footage (see README.md, "Key finding #2") -- the two
genuine photos of the same person scored WORSE (less similar) than over
a third of random other people's crops.

A network trained on ImageNet, even for an unrelated task (1000-way object
classification, nothing to do with re-identifying people), still learns
general-purpose visual features in its earlier/middle layers -- edges,
textures, color patterns, coarse shape -- that turn out to be far more
lighting- and pose-invariant than a raw color histogram, because the
network had to learn to recognize the same object class across wildly
different lighting/angle/background conditions during training. Taking
its features from just before the final classification layer (rather
than the 1000-way class scores themselves, which are specialized for
distinguishing unrelated object categories) repurposes that general
visual understanding as an embedding for a completely different
comparison task. This is a well-known technique ("transfer learning" /
using a pretrained network as a fixed feature extractor) and is a huge
step up from hand-built color histograms without requiring any
person-specific training data.

A real person re-identification network (OSNet, TransReID, etc.), trained
specifically to recognize the same person across cameras/lighting/pose,
would do even better than a generic ImageNet classifier repurposed this
way -- but those aren't available as plain downloadable files through the
network access this environment allows (HuggingFace and PyTorch's own CDN
are both unreachable here). ResNet18 was chosen because an actual ONNX
binary of it is downloadable as a plain GitHub release asset (release
assets bypass git-lfs entirely, unlike files stored directly in a repo,
which is what made earlier attempts at other model zoos fail -- see the
conversation this pipeline was built in for that whole story). This is a
genuine, working upgrade over the color histogram, not a perfect one.

Usage:
    python setup_reid_model.py
Produces: models/resnet18_embedding.onnx
"""

import os

import onnx
import urllib.request

RESNET18_URL = "https://github.com/shoz-f/axon_interp/releases/download/0.0.1/resnet18.onnx"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RAW_PATH = os.path.join(MODELS_DIR, "resnet18.onnx")
EMBEDDING_PATH = os.path.join(MODELS_DIR, "resnet18_embedding.onnx")

# Found by inspecting the model's graph: the GlobalAveragePool node right
# before the final Flatten + Gemm (fully-connected classifier) layers.
# If a different ResNet18 ONNX export is used instead, find the equivalent
# node with: `python3 -c "import onnx; m=onnx.load('resnet18.onnx');
# [print(n.op_type, n.output) for n in m.graph.node]"` and look for the
# GlobalAveragePool node's output name.
EMBEDDING_TENSOR_NAME = "onnx::Flatten_189"


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(RAW_PATH):
        print(
            f"Downloading ResNet18 (ImageNet-pretrained, ~45MB) from {RESNET18_URL} ...")
        urllib.request.urlretrieve(RESNET18_URL, RAW_PATH)
    else:
        print(f"{RAW_PATH} already exists, skipping download.")

    print("Adding the pre-classification feature vector as a model output...")
    model = onnx.load(RAW_PATH)
    embedding_output = onnx.helper.make_tensor_value_info(
        EMBEDDING_TENSOR_NAME, onnx.TensorProto.FLOAT, [1, 512, 1, 1],
    )
    model.graph.output.append(embedding_output)
    onnx.save(model, EMBEDDING_PATH)
    print(f"Saved {EMBEDDING_PATH}")


if __name__ == "__main__":
    main()
