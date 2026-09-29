"""
label_crops.py
----------------
Turns "I don't have many photos of the staff member" into "I have
hundreds of them" -- by pulling candidate person crops directly out of
sample.mp4 using the SAME detector already used everywhere else in this
project, instead of requiring separate reference photos. A ~53-second
clip has ~1,300 frames, each with ~15-18 people in it; even sampling
every few frames gives thousands of candidate crops to choose from.

Two steps, run separately so the slow part (detection) only happens once:

    python3 label_crops.py extract --video sample.mp4 --out_dir training_data --stride 5
    python3 label_crops.py label --out_dir training_data

`extract` runs the detector across the video and saves every detected
person as a small image file plus a manifest -- this is the slow step
(same ~3.9-7.4s/frame cost as everything else in this project that uses
the full detector). `label` opens a simple Tkinter window (built into
Python, no extra install) showing one crop at a time -- press Y if it's
the staff member, N if it's not, S to skip if you're unsure. Progress
saves after every label, so you can stop and resume anytime; you do NOT
need to label every extracted crop, just enough of each class (see the
running counts shown in the label window).

What you actually need, roughly:
  - At least ~20-30 labeled as the staff member, covering DIFFERENT
    moments in the clip (different poses/angles/lighting) -- labeling 30
    crops from one 2-second stretch is much less useful than 30 crops
    spread across the whole video, since they'd nearly all look alike.
  - At least ~50-100 labeled as NOT the staff member, ideally covering
    many DIFFERENT other people, not just one or two repeatedly -- this
    is what teaches the model what to tell the staff member apart FROM.
  - More is better, but a lopsided ratio (thousands of one class, a
    handful of the other) doesn't help much -- balance matters more than
    raw count in a small dataset.
"""

import argparse
import json
import os

import cv2

from detector import PersonDetector


def extract(video_path, out_dir, model, rotations, conf_thresh, stride):
    os.makedirs(out_dir, exist_ok=True)
    candidates_dir = os.path.join(out_dir, "candidates")
    os.makedirs(candidates_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "candidates_manifest.json")

    manifest = json.load(open(manifest_path)) if os.path.exists(
        manifest_path) else []
    done_frames = {m["frame_index"] for m in manifest}

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    if model == "tiny":
        cfg, weights = "yolov4-tiny.cfg", "yolov4-tiny.weights"
    else:
        cfg, weights = "yolov4.cfg", "yolov4.weights"
    detector = PersonDetector(
        os.path.join(model_dir, cfg), os.path.join(model_dir, weights),
        os.path.join(model_dir, "coco.names"),
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride != 0 or frame_idx in done_frames:
            frame_idx += 1
            continue

        detections = detector.detect(
            frame, conf_thresh=conf_thresh, rotations=rotations)
        for i, det in enumerate(detections):
            x, y, w, h = det["bbox"]
            x, y = max(0, x), max(0, y)
            w, h = max(1, min(w, frame.shape[1] - x)
                       ), max(1, min(h, frame.shape[0] - y))
            crop = frame[y:y + h, x:x + w]
            if crop.size == 0:
                continue
            crop_name = f"{frame_idx:06d}_{i:02d}.png"
            cv2.imwrite(os.path.join(candidates_dir, crop_name), crop)
            manifest.append({
                "frame_index": frame_idx,
                "bbox_xywh": det["bbox"],
                "crop_path": os.path.join("candidates", crop_name),
                "label": None,  # filled in by the `label` step
            })
            saved += 1

        if frame_idx % (stride * 20) == 0:
            print(
                f"...frame {frame_idx}/{total_frames}, {saved} crops saved so far")
            json.dump(manifest, open(manifest_path, "w"), indent=2)

        frame_idx += 1

    cap.release()
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"\nDone. {saved} candidate crops saved to {candidates_dir}/")
    print(f"Now run: python3 label_crops.py label --out_dir {out_dir}")


def label_gui(out_dir):
    import tkinter as tk
    from PIL import Image, ImageTk  # pip install pillow

    manifest_path = os.path.join(out_dir, "candidates_manifest.json")
    manifest = json.load(open(manifest_path))
    unlabeled = [m for m in manifest if m["label"] is None]
    if not unlabeled:
        print("Everything in the manifest is already labeled.")
        return

    n_staff = sum(1 for m in manifest if m["label"] == "staff")
    n_not = sum(1 for m in manifest if m["label"] == "not_staff")

    root = tk.Tk()
    root.title(
        "Label crops -- Y = staff, N = not staff, S = skip, Esc = save & quit")

    img_label = tk.Label(root)
    img_label.pack()
    status = tk.Label(root, font=("Arial", 12))
    status.pack()
    hint = tk.Label(root, text="Y = staff member   |   N = not staff   |   S = skip/unsure   |   Esc = save & quit",
                    font=("Arial", 10))
    hint.pack()

    state = {"i": 0}

    def show_current():
        if state["i"] >= len(unlabeled):
            status.config(text="All done!")
            img_label.config(image="")
            return
        m = unlabeled[state["i"]]
        img = Image.open(os.path.join(out_dir, m["crop_path"]))
        img.thumbnail((300, 500))
        photo = ImageTk.PhotoImage(img)
        img_label.config(image=photo)
        img_label.image = photo  # keep a reference so it isn't garbage-collected
        status.config(text=f"crop {state['i'] + 1}/{len(unlabeled)}  |  "
                      f"labeled so far: {n_staff} staff, {n_not} not_staff  |  "
                      f"frame {m['frame_index']}")

    def save_manifest():
        json.dump(manifest, open(manifest_path, "w"), indent=2)

    def label_current(value):
        nonlocal n_staff, n_not
        if state["i"] < len(unlabeled):
            unlabeled[state["i"]]["label"] = value
            if value == "staff":
                n_staff += 1
            elif value == "not_staff":
                n_not += 1
            state["i"] += 1
            if state["i"] % 10 == 0:
                save_manifest()
            show_current()

    def on_key(event):
        key = event.keysym.lower()
        if key == "y":
            label_current("staff")
        elif key == "n":
            label_current("not_staff")
        elif key == "s":
            state["i"] += 1
            show_current()
        elif key == "escape":
            save_manifest()
            print(
                f"Saved. {n_staff} labeled staff, {n_not} labeled not_staff.")
            root.destroy()

    root.bind("<Key>", on_key)
    show_current()
    root.mainloop()
    save_manifest()


def review_gui(out_dir, review_class):
    """
    Shows every crop CURRENTLY labeled `review_class` (default "staff"),
    one at a time, so mislabeled crops can be fixed before training --
    label_crops.py's `label` mode only ever shows UNLABELED crops, so this
    is the only way to revisit a decision already made.

    Keys:
      K (keep)        -- current label is correct, move to next.
      F (flip)        -- wrong class -- flips staff <-> not_staff.
      U (unlabel)      -- neither/unclear/bad crop -- clears the label
                          back to None (returns it to the unlabeled pool
                          for label_crops.py's `label` mode to show again).
      Left arrow / B   -- go back to the previous crop (in case of a
                          misclick -- review sessions are usually short
                          enough that stepping back and forth is fine).
      Esc              -- save and quit.
    Saves after every single change (unlike `label`'s every-10 batching)
    since a review session is typically short and correctness here
    matters more than minimizing disk writes.
    """
    import tkinter as tk
    from PIL import Image, ImageTk

    manifest_path = os.path.join(out_dir, "candidates_manifest.json")
    manifest = json.load(open(manifest_path))
    reviewing = [m for m in manifest if m["label"] == review_class]
    if not reviewing:
        print(f"No crops currently labeled '{review_class}' to review.")
        return

    root = tk.Tk()
    root.title(f"Review '{review_class}' crops -- K=keep, F=flip, U=unlabel, "
               f"<-/B=back, Esc=save & quit")

    img_label = tk.Label(root)
    img_label.pack()
    status = tk.Label(root, font=("Arial", 12))
    status.pack()
    hint = tk.Label(
        root,
        text="K = keep (correct)   |   F = flip to other class   |   U = clear label   |   "
             "<- / B = back   |   Esc = save & quit",
        font=("Arial", 10),
    )
    hint.pack()

    state = {"i": 0}
    other_class = "not_staff" if review_class == "staff" else "staff"

    def save_manifest():
        json.dump(manifest, open(manifest_path, "w"), indent=2)

    def show_current():
        if state["i"] >= len(reviewing):
            status.config(text="Review complete!")
            img_label.config(image="")
            return
        if state["i"] < 0:
            state["i"] = 0
        m = reviewing[state["i"]]
        img = Image.open(os.path.join(out_dir, m["crop_path"]))
        img.thumbnail((300, 500))
        photo = ImageTk.PhotoImage(img)
        img_label.config(image=photo)
        img_label.image = photo
        status.config(text=f"crop {state['i'] + 1}/{len(reviewing)}  |  "
                      f"currently labeled: {m['label']}  |  frame {m['frame_index']}")

    def go_next():
        state["i"] += 1
        show_current()

    def on_key(event):
        key = event.keysym.lower()
        if state["i"] >= len(reviewing):
            if key == "escape":
                save_manifest()
                root.destroy()
            return
        m = reviewing[state["i"]]
        if key == "k":
            go_next()
        elif key == "f":
            m["label"] = other_class
            save_manifest()
            go_next()
        elif key == "u":
            m["label"] = None
            save_manifest()
            go_next()
        elif key in ("left", "b"):
            state["i"] -= 1
            show_current()
        elif key == "escape":
            save_manifest()
            print(
                f"Saved. Reviewed up to crop {state['i'] + 1}/{len(reviewing)}.")
            root.destroy()

    root.bind("<Key>", on_key)
    show_current()
    root.mainloop()
    save_manifest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--video", default="sample.mp4")
    p_extract.add_argument("--out_dir", default="training_data")
    p_extract.add_argument("--model", choices=["tiny", "full"], default="full")
    p_extract.add_argument("--rotations", default="0,180")
    p_extract.add_argument("--conf_thresh", type=float, default=0.25)
    p_extract.add_argument("--stride", type=int, default=5,
                           help="extract candidates from every Nth frame")

    p_label = sub.add_parser("label")
    p_label.add_argument("--out_dir", default="training_data")

    p_review = sub.add_parser("review")
    p_review.add_argument("--out_dir", default="training_data")
    p_review.add_argument("--class", dest="review_class", choices=["staff", "not_staff"],
                          default="staff", help="which labeled class to review")

    args = parser.parse_args()
    if args.mode == "extract":
        rotations = tuple(int(x) for x in args.rotations.split(","))
        extract(args.video, args.out_dir, args.model,
                rotations, args.conf_thresh, args.stride)
    elif args.mode == "label":
        label_gui(args.out_dir)
    else:
        review_gui(args.out_dir, args.review_class)
