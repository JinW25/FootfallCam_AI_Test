"""
single_frame.py
------------------
Runs the pipeline on ONE frame and saves every intermediate step as its
own image, for inspection/debugging/explanation:

    01_original_frame.png
    02_rotated_<deg>deg.png              (one per rotation angle used)
    03_detections_rot_<deg>deg.png       (raw per-rotation detections, in
                                           that rotation's own coordinate
                                           space -- exactly what the
                                           network saw and found there)
    04_merged_detections.png             (all detections mapped back to
                                           the original frame, after NMS --
                                           the winner highlighted in red)
    05_winner_crop_raw.png               (the winning crop, straight out
                                           of the original frame)
    06_winner_crop_resized_224.png       (after BGR->RGB + resize to 224x224,
                                           the network's expected input size)
    07_winner_crop_normalized_visual.png (a VISUALIZATION of the normalized
                                           tensor actually fed to the network
                                           -- rescaled back into a viewable
                                           0-255 range for display; the real
                                           values are NOT in this range, see
                                           the printed stats)
    08_comparison_vs_reference.png       (winner crop next to the reference
                                           gallery image(s), with the actual
                                           similarity score)

Every number shown is computed by calling the real pipeline code
(detector.py's own internal rotation/detection methods, reid.py's own
embed()/similarity()) -- nothing here is a separate re-implementation
that could quietly drift from what main.py actually does.

Usage:
    python3 single_frame.py --video sample.mp4 --frame 700 --out_dir frame_700_steps
"""

import argparse
import os

import cv2
import numpy as np

from detector import PersonDetector
from reid import embed, best_match, similarity, load_reference_gallery, _MEAN, _STD, _INPUT_SIZE


def read_frame_sequential(video_path, frame_idx):
    """Sequential read up to the target frame -- never seeks, so this is
    always frame-exact (see module docstring in the original version for
    why cv2.CAP_PROP_POS_FRAMES seeking is a known source of inaccuracy
    on compressed video)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames:
        raise ValueError(
            f"frame {frame_idx} is out of range (video has {total_frames} frames)")
    frame = None
    for i in range(frame_idx + 1):
        ok, f = cap.read()
        if not ok:
            raise RuntimeError(f"video ended early, at frame {i}")
        if i == frame_idx:
            frame = f
    cap.release()
    return frame, fps


def run(video_path, frame_idx, ref_dir, model, rotations, conf_thresh, average_refs, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    if model == "tiny":
        cfg, weights = "yolov4-tiny.cfg", "yolov4-tiny.weights"
    else:
        cfg, weights = "yolov4.cfg", "yolov4.weights"
    detector = PersonDetector(
        os.path.join(model_dir, cfg), os.path.join(model_dir, weights),
        os.path.join(model_dir, "coco.names"),
    )
    gallery = load_reference_gallery(ref_dir, average=average_refs)
    print(f"Loaded reference gallery: "
          f"{'1 averaged embedding' if average_refs else f'{len(gallery)} separate embeddings'} "
          f"from {ref_dir}/")

    frame, fps = read_frame_sequential(video_path, frame_idx)
    print(f"\nFrame {frame_idx} (t={frame_idx/fps:.2f}s), shape={frame.shape}")

    # ---------- Step 1: original frame ----------
    cv2.imwrite(os.path.join(out_dir, "01_original_frame.png"), frame)

    # ---------- Steps 2-3: per-rotation views and their raw detections ----------
    # Mirrors detector.py's detect() internals exactly, one rotation at a time,
    # so we can save what each individual pass sees before merging.
    all_boxes, all_confs = [], []
    for deg in rotations:
        k = (deg // 90) % 4
        rotated = np.rot90(frame, k) if k else frame
        rh, rw = rotated.shape[:2]

        cv2.imwrite(os.path.join(out_dir, f"02_rotated_{deg}deg.png"), rotated)

        boxes, confs = detector._detect_single_orientation(
            rotated, conf_thresh)
        print(
            f"  rotation {deg:>3}deg: {len(boxes)} raw candidate(s) before NMS")

        vis = rotated.copy()
        for (x, y, w, h), c in zip(boxes, confs):
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
            cv2.putText(vis, f"{c:.2f}", (x, max(0, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(vis, f"rotation={deg}deg, {len(boxes)} raw candidates (pre-NMS)",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(
            out_dir, f"03_detections_rot_{deg}deg.png"), vis)

        for box in boxes:
            all_boxes.append(detector._rotate_box_to_original(box, k, rw, rh))
        all_confs.extend(confs)

    # ---------- Step 4: merged, post-NMS detections mapped back to the original frame ----------
    idxs = cv2.dnn.NMSBoxes(all_boxes, all_confs,
                            conf_thresh, 0.45) if all_boxes else []
    idxs = idxs.flatten() if len(idxs) else []
    detections = [{"bbox": all_boxes[i], "conf": all_confs[i]} for i in idxs]
    print(f"\n{len(detections)} people after merging all rotations + NMS")

    h_img, w_img = frame.shape[:2]
    results = []
    for det in detections:
        x, y, w, h = det["bbox"]
        x, y = max(0, x), max(0, y)
        w, h = max(1, min(w, w_img - x)), max(1, min(h, h_img - y))
        crop = frame[y:y + h, x:x + w]
        emb = embed(crop)
        sim, _ = best_match(emb, gallery)
        results.append(
            {"bbox": [x, y, w, h], "similarity": sim, "embedding": emb})
    results.sort(key=lambda r: -r["similarity"])

    print()
    for i, r in enumerate(results):
        marker = "  <-- BEST" if i == 0 else ""
        print(
            f"  #{i+1}: bbox={r['bbox']} similarity={r['similarity']:.4f}{marker}")

    vis = frame.copy()
    for i, r in enumerate(results):
        x, y, w, h = r["bbox"]
        color = (0, 0, 255) if i == 0 else (150, 150, 150)
        thickness = 2 if i == 0 else 1
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(vis, f"{r['similarity']:.3f}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.putText(vis, f"frame={frame_idx} t={frame_idx/fps:.2f}s -- {len(results)} merged, post-NMS",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(out_dir, "04_merged_detections.png"), vis)

    if not results:
        print("No detections at all -- stopping here.")
        return results

    winner = results[0]
    x, y, w, h = winner["bbox"]

    # ---------- Step 5: the winning crop, raw ----------
    raw_crop = frame[y:y + h, x:x + w]
    cv2.imwrite(os.path.join(out_dir, "05_winner_crop_raw.png"), raw_crop)

    # ---------- Step 6: BGR->RGB + resize to network input size ----------
    # Mirrors reid.py's embed() preprocessing exactly, stopping right before
    # normalization so this step is still directly viewable as a normal image.
    rgb = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    resized = cv2.resize(rgb, (_INPUT_SIZE, _INPUT_SIZE))
    resized_viewable = cv2.cvtColor(
        (resized * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(
        out_dir, "06_winner_crop_resized_224.png"), resized_viewable)

    # ---------- Step 7: normalization (visualized) ----------
    normalized = (resized - _MEAN) / _STD
    print(f"\nNormalized tensor stats (fed to the network, NOT a viewable image on its own): "
          f"min={normalized.min():.3f}, max={normalized.max():.3f}, mean={normalized.mean():.3f}")
    # Rescale back into 0-255 purely for illustration, so the effect of
    # normalization (contrast/brightness shift per channel) is visible --
    # this rescaled version is NOT what the network actually receives.
    vis_norm = normalized.copy()
    vis_norm = (vis_norm - vis_norm.min()) / \
        (vis_norm.max() - vis_norm.min() + 1e-8)
    vis_norm = cv2.cvtColor(
        (vis_norm * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.putText(vis_norm, "illustrative only -- see printed stats for real values", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(
        out_dir, "07_winner_crop_normalized_visual.png"), vis_norm)

    # ---------- Step 8: side-by-side comparison against the reference gallery ----------
    ref_paths = [os.path.join(ref_dir, f) for f in sorted(os.listdir(ref_dir))
                 if os.path.exists(os.path.join(ref_dir, f))]
    target_h = 220
    panels = []
    winner_panel = cv2.resize(
        raw_crop, (int(raw_crop.shape[1] * target_h / raw_crop.shape[0]), target_h))
    cv2.putText(winner_panel, "winner", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    panels.append(winner_panel)
    for p in ref_paths:
        ref_img = cv2.imread(p)
        if ref_img is None:
            continue
        panel = cv2.resize(
            ref_img, (int(ref_img.shape[1] * target_h / ref_img.shape[0]), target_h))
        cv2.putText(panel, os.path.basename(p), (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        panels.append(panel)
    sep = np.full((target_h, 12, 3), 255, dtype=np.uint8)
    combined = panels[0]
    for p in panels[1:]:
        combined = cv2.hconcat([combined, sep, p])
    banner = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(banner, f"similarity vs reference gallery = {winner['similarity']:.4f}",
                (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
    combined = cv2.vconcat([banner, combined])
    cv2.imwrite(os.path.join(
        out_dir, "08_comparison_vs_reference.png"), combined)

    print(f"\nAll intermediate step images saved to {out_dir}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="sample.mp4")
    parser.add_argument("--frame", type=int, required=True,
                        help="frame index to inspect")
    parser.add_argument("--ref_dir", default="reference_crops")
    parser.add_argument("--model", choices=["tiny", "full"], default="full")
    parser.add_argument("--rotations", default="0,90,180,270")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--no_average_refs", action="store_true",
                        help="keep multiple reference photos as separate embeddings "
                        "instead of averaging them into one (see reid.py)")
    parser.add_argument("--out_dir", default=None,
                        help="directory for all step images (default: frame_<N>_steps/)")
    args = parser.parse_args()

    rotations = tuple(int(x) for x in args.rotations.split(","))
    out_dir = args.out_dir or f"frame_{args.frame}_steps"
    run(args.video, args.frame, args.ref_dir, args.model, rotations, args.conf_thresh,
        not args.no_average_refs, out_dir)
