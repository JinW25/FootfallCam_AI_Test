# Staff Identification Pipeline

## Setup

```bash
pip install opencv-python-headless numpy scipy onnx onnxruntime pillow

mkdir -p models
curl -L -o models/yolov4.cfg      https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
curl -L -o models/yolov4.weights  https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights
curl -L -o models/coco.names      https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

python setup_reid_model.py
```

Put your video at `sample.mp4` in the project root, and reference photos
of the staff member in `reference_crops/`.

## Running main.py

```bash
python main.py --video sample.mp4 --out_dir output --stride 1
```

Produces `output/results.json` (per-frame presence, bounding box, pixel
coordinates) and `output/annotated_output.mp4`.

Common options:

| Flag | Default | Meaning |
|---|---|---|
| `--stride` | `1` | Process every Nth frame (higher = faster, less thorough) |
| `--model` | `full` | `full` or `tiny` YOLOv4 |
| `--rotations` | `0,180` | Degrees to check, e.g. `0,90,180,270` |
| `--confirm_threshold` | `0.90` | Similarity a frame must clear to count |
| `--min_streak` | `3` | Consecutive frames required to trust a match |
| `--no_average_refs` | off | Keep multiple reference photos separate instead of averaging them into one |

To use a fine-tuned model instead of the default one:

```bash
export REID_MODEL_PATH=models/finetuned_reid.onnx   # macOS/Linux
set REID_MODEL_PATH=models\finetuned_reid.onnx        # Windows
```

## Labeling data

Pulls candidate crops directly from the video so you can label them:

```bash
python label_crops.py extract --video sample.mp4 --out_dir training_data --stride 5
python label_crops.py label --out_dir training_data
```

In the label window: **Y** = staff, **N** = not staff, **S** = skip.

To review and fix labels afterward:

```bash
python label_crops.py review --out_dir training_data --class staff
```

**K** = keep, **F** = flip to the other class, **U** = clear the label,
**←** = back, **Esc** = save and quit.

Once labeled, train a fine-tuned model:

```bash
pip install torch torchvision
python train_reid_head.py --data_dir training_data --epochs 100
```