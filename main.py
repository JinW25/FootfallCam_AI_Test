"""
main.py
-------
Single-pass pipeline: for every sampled frame, detect everyone, embed and
score each detected crop against the reference gallery, and record all of
it. Then decide who the staff member is and when they're present using
CONFIRM-THEN-BRIDGE:

  1. CONFIRM: a frame only counts as a genuine sighting if its single
     best-scoring detection clears CONFIRM_THRESHOLD, AND at least
     MIN_STREAK consecutive sampled frames in a row do the same. One lucky
     high-scoring frame is not enough on its own. This step never looks at
     position, tracking, or identity continuity at all -- it only trusts
     the appearance match, frame by frame, independently.

  2. BRIDGE: short gaps between two confirmed streaks get filled in,
     preferring an actual (if sub-threshold) detection near the expected
     position over pure geometric interpolation, but only when the gap is
     short enough in both time and distance to be a plausible continuous
     human movement.


Usage:
    python main.py --video sample.mp4 --out_dir output --stride 1

"""

import argparse
import json
import os

import cv2

from detector import PersonDetector
from reid import embed, best_match, load_reference_gallery

STAFF_COLOR = (0, 0, 255)
# a gap longer than this between two confirmed streaks
BRIDGE_MAX_GAP_SECONDS = 2.0
# is never bridged, however close in position -- too long a gap to trust regardless.
# how far a person could plausibly move during a gap;
BRIDGE_MAX_SPEED_PX_PER_SEC = 250
# scales with the gap's actual duration, not a fixed pixel budget.
# how close a sub-threshold detection must be to the
BRIDGE_NEARBY_PX = 60
# interpolated position to be used as-is instead of falling back to pure interpolation.


def detect_and_score(video_path, ref_dir, out_dir, model, rotations, conf_thresh,
                     stride, average_refs):
    """
    Single pass over the video: detect every person in every sampled
    frame, embed and score each against the reference gallery. Every raw
    frame is cached to disk (for the render step later) regardless of
    stride; only sampled frames get the (expensive) detector run on them.

    Returns (detections_by_frame, fps, width, height, total_frames,
    frame_cache_dir) where detections_by_frame is {frame_idx: [detections]},
    each detection a dict with "bbox_xywh", "xy_center", "similarity".
    """
    os.makedirs(out_dir, exist_ok=True)
    frame_cache_dir = os.path.join(out_dir, "_frame_cache")
    os.makedirs(frame_cache_dir, exist_ok=True)

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

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    detections_by_frame = {}
    frame_idx = 0
    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(frame_cache_dir, f"{frame_idx:06d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

        if frame_idx % stride == 0:
            raw_detections = detector.detect(
                frame, conf_thresh=conf_thresh, rotations=rotations)
            scored = []
            for det in raw_detections:
                x, y, w, h = det["bbox"]
                x, y = max(0, x), max(0, y)
                w, h = max(1, min(w, width - x)), max(1, min(h, height - y))
                crop = frame[y:y + h, x:x + w]
                sim, _ = best_match(embed(crop), gallery)
                scored.append({
                    "bbox_xywh": [x, y, w, h],
                    "xy_center": [int(x + w / 2), int(y + h / 2)],
                    "similarity": round(sim, 4),
                })
            detections_by_frame[frame_idx] = scored

            processed += 1
            if processed % 20 == 0:
                print(
                    f"...processed {processed} sampled frames (video frame {frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    return detections_by_frame, fps, width, height, total_frames, frame_cache_dir


def compute_best_per_frame(detections_by_frame, confirm_threshold):
    """For every sampled frame, the single best-scoring detection and
    whether it clears confirm_threshold. Purely independent, frame by
    frame -- no state carried between frames, no position assist."""
    best = {}
    for frame_idx, dets in detections_by_frame.items():
        if not dets:
            best[frame_idx] = {"bbox_xywh": None,
                               "similarity": None, "confirmed": False}
            continue
        top = max(dets, key=lambda d: d["similarity"])
        best[frame_idx] = {
            "bbox_xywh": top["bbox_xywh"],
            "similarity": top["similarity"],
            "confirmed": top["similarity"] >= confirm_threshold,
        }
    return best


def find_confirmed_streaks(best_per_frame, stride, min_streak):
    """Runs of >= min_streak CONSECUTIVE sampled frames (no missed sample
    in between) that all independently cleared confirm_threshold."""
    confirmed_frames = sorted(
        f for f, e in best_per_frame.items() if e["confirmed"])
    streaks, current = [], []
    for f in confirmed_frames:
        if current and f - current[-1] > stride:
            if len(current) >= min_streak:
                streaks.append(current)
            current = []
        current.append(f)
    if len(current) >= min_streak:
        streaks.append(current)
    return streaks


# a fixed pixel allowance for ordinary bounding-box
JUMP_NOISE_FLOOR_PX = 50
# jitter (slightly different crop each frame, small pose shifts) -- this does NOT
# shrink as the time gap shrinks, unlike real movement, because detection noise is
# roughly constant regardless of how close in time two samples are. This is exactly
# what an earlier, reverted attempt at this got wrong: a pure BRIDGE_MAX_SPEED_PX_PER_SEC
# * gap_seconds budget goes to ~10px at stride=1 (0.04s apart), which is smaller than
# ordinary jitter, and shattered every streak. Adding this floor fixes that.
# on top of the floor, the same real-movement budget
JUMP_MAX_SPEED_PX_PER_SEC = 250
# used everywhere else in this file (bridge_gaps' gap-filling check).


def split_far_jumps(streaks, best_per_frame, fps, min_streak):
    """
    A "confirmed" streak is built purely from appearance (compute_best_per_frame
    never looks at position at all) -- so it's possible, if a different person
    momentarily scores even higher than the real target in one specific frame,
    for a streak to silently contain a jump to a different physical person,
    coincidentally strung together because both happened to independently
    clear the threshold. This checks every pair of ADJACENT frames within
    each already-formed streak using CENTROID distance (box center, not
    raw corners), and splits the streak wherever the jump exceeds
    JUMP_NOISE_FLOOR_PX + JUMP_MAX_SPEED_PX_PER_SEC * elapsed_seconds --
    a budget that stays sensible at both very short (stride=1) and longer
    gaps, unlike a pure linear-speed formula. Each resulting piece is then
    re-checked against min_streak; an isolated jump-to-a-stranger is
    typically too short on its own to survive and gets discarded, while the
    genuine trajectory on either side is preserved (and bridge_gaps can
    often reconnect the two pieces afterward anyway).

    Deliberately a POST-HOC check on already-confirmed streaks, never a
    bias applied during confirmation itself -- see README.md for why an
    earlier attempt that biased confirmation directly caused a regression.
    """
    result = []
    for streak in streaks:
        piece = [streak[0]]
        for prev_f, f in zip(streak, streak[1:]):
            prev_box = best_per_frame[prev_f]["bbox_xywh"]
            box = best_per_frame[f]["bbox_xywh"]
            prev_c = (prev_box[0] + prev_box[2] / 2,
                      prev_box[1] + prev_box[3] / 2)
            c = (box[0] + box[2] / 2, box[1] + box[3] / 2)
            gap_seconds = (f - prev_f) / fps
            dist = ((c[0] - prev_c[0]) ** 2 + (c[1] - prev_c[1]) ** 2) ** 0.5
            max_dist = JUMP_NOISE_FLOOR_PX + JUMP_MAX_SPEED_PX_PER_SEC * gap_seconds
            if dist > max_dist:
                if len(piece) >= min_streak:
                    result.append(piece)
                piece = [f]
            else:
                piece.append(f)
        if len(piece) >= min_streak:
            result.append(piece)
    return result


def bridge_gaps(streaks, best_per_frame, detections_by_frame, fps):
    """
    Fills the gap between each pair of temporally-adjacent confirmed
    streaks, IF the gap is short enough in both time and position to be a
    plausible continuous human movement. Returns {frame_idx: (bbox, status,
    similarity)} covering every confirmed and successfully-bridged frame
    only. similarity is the real measured score for "confirmed" frames and
    for bridged frames that used a real nearby detection; None for frames
    filled by pure geometric interpolation, since there's no actual
    measurement to report there -- it's a position guess, not a match.
    """
    result = {}
    for streak in streaks:
        for f in streak:
            result[f] = (best_per_frame[f]["bbox_xywh"],
                         "confirmed", best_per_frame[f]["similarity"])

    streaks_sorted = sorted(streaks, key=lambda s: s[0])
    for i in range(len(streaks_sorted) - 1):
        end_frame, start_frame = streaks_sorted[i][-1], streaks_sorted[i + 1][0]
        gap_seconds = (start_frame - end_frame) / fps
        if gap_seconds > BRIDGE_MAX_GAP_SECONDS:
            continue

        end_box = best_per_frame[end_frame]["bbox_xywh"]
        start_box = best_per_frame[start_frame]["bbox_xywh"]
        end_c = (end_box[0] + end_box[2] / 2, end_box[1] + end_box[3] / 2)
        start_c = (start_box[0] + start_box[2] / 2,
                   start_box[1] + start_box[3] / 2)
        total_dist = ((start_c[0] - end_c[0]) ** 2 +
                      (start_c[1] - end_c[1]) ** 2) ** 0.5
        if total_dist > BRIDGE_MAX_SPEED_PX_PER_SEC * gap_seconds:
            continue

        for f in sorted(fr for fr in detections_by_frame if end_frame < fr < start_frame):
            if not detections_by_frame[f]:
                # Zero people detected in this frame at all -- don't fabricate a box
                # from pure geometry with no underlying evidence whatsoever. This
                # matches confirm_and_bridge.py's behavior (which can never even
                # attempt to bridge such a frame, since it's simply absent from
                # all_detections.json when nobody was detected).
                continue
            t = (f - end_frame) / (start_frame - end_frame)
            interp_c = (end_c[0] + t * (start_c[0] - end_c[0]),
                        end_c[1] + t * (start_c[1] - end_c[1]))

            best_real, best_dist = None, None
            for d in detections_by_frame[f]:
                c = d["xy_center"]
                dist = ((c[0] - interp_c[0]) ** 2 +
                        (c[1] - interp_c[1]) ** 2) ** 0.5
                if best_dist is None or dist < best_dist:
                    best_real, best_dist = d, dist
            if best_real is not None and best_dist <= BRIDGE_NEARBY_PX:
                result[f] = (best_real["bbox_xywh"], "bridged",
                             best_real["similarity"])
            else:
                w = end_box[2] + t * (start_box[2] - end_box[2])
                h = end_box[3] + t * (start_box[3] - end_box[3])
                bbox = [int(interp_c[0] - w / 2),
                        int(interp_c[1] - h / 2), int(w), int(h)]
                result[f] = (bbox, "bridged", None)

    return result


def render_and_save(out_dir, frame_cache_dir, sampled_frames, staff_by_frame, fps, stride, total_frames):
    out_video_path = os.path.join(out_dir, "annotated_output.mp4")
    first = cv2.imread(os.path.join(
        frame_cache_dir, f"{sampled_frames[0]:06d}.jpg"))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps / stride, (w, h))

    results = []
    for frame_idx in sampled_frames:
        img = cv2.imread(os.path.join(frame_cache_dir, f"{frame_idx:06d}.jpg"))
        if img is None:
            continue
        entry = staff_by_frame.get(frame_idx)
        results.append({
            "frame_index": frame_idx,
            "timestamp_sec": round(frame_idx / fps, 2),
            "staff_present": entry is not None,
            "staff_bbox_xywh": entry[0] if entry else None,
            "staff_xy_center": (
                [int(entry[0][0] + entry[0][2] / 2),
                 int(entry[0][1] + entry[0][3] / 2)]
                if entry else None
            ),
            "status": entry[1] if entry else None,  # "confirmed" or "bridged"
            "similarity": (round(entry[2], 4) if entry and entry[2] is not None else None),
            # real measured similarity for "confirmed" frames and for "bridged" frames that
            # used a real nearby detection; None for pure-interpolation bridged frames, since
            # there's no actual match to report there -- it's a position guess, not a score.
        })
        if entry is not None:
            x, y, bw, bh = entry[0]
            cx, cy = x + bw // 2, y + bh // 2
            cv2.rectangle(img, (x, y), (x + bw, y + bh), STAFF_COLOR, 2)
            sim_text = f"STAFF sim={entry[2]:.2f}" if entry[2] is not None else "STAFF"
            cv2.putText(img, sim_text, (x, max(0, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, STAFF_COLOR, 2)
            cv2.putText(img, f"({cx},{cy})", (x, y + bh + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, STAFF_COLOR, 1)
        cv2.putText(img, f"t={frame_idx/fps:.1f}s frame={frame_idx}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        writer.write(img)
    writer.release()

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results, out_video_path


def run(video_path, ref_dir, out_dir, model="full", rotations=(0, 180), stride=1,
        conf_thresh=0.25, average_refs=True, confirm_threshold=0.90, min_streak=3):
    detections_by_frame, fps, width, height, total_frames, frame_cache_dir = detect_and_score(
        video_path, ref_dir, out_dir, model, rotations, conf_thresh, stride, average_refs)

    best_per_frame = compute_best_per_frame(
        detections_by_frame, confirm_threshold)
    streaks = find_confirmed_streaks(best_per_frame, stride, min_streak)
    print(f"\n{len(streaks)} confirmed streaks found "
          f"(>= {min_streak} consecutive sampled frames clearing {confirm_threshold}):")
    for s in streaks:
        print(
            f"  frames {s[0]}-{s[-1]} (t={s[0]/fps:.2f}s-{s[-1]/fps:.2f}s), {len(s)} samples")

    streaks = split_far_jumps(streaks, best_per_frame, fps, min_streak)
    print(f"\nAfter splitting out far centroid jumps (a different person momentarily "
          f"scoring higher mid-streak): {len(streaks)} streaks remain:")
    for s in streaks:
        print(
            f"  frames {s[0]}-{s[-1]} (t={s[0]/fps:.2f}s-{s[-1]/fps:.2f}s), {len(s)} samples")

    staff_by_frame = bridge_gaps(
        streaks, best_per_frame, detections_by_frame, fps)
    n_confirmed = sum(
        1 for _, status, _ in staff_by_frame.values() if status == "confirmed")
    n_bridged = sum(1 for _, status, _ in staff_by_frame.values()
                    if status == "bridged")
    print(f"\n{n_confirmed} confirmed frames, {n_bridged} bridged frames, "
          f"{len(staff_by_frame)} total frames with a staff position.")

    sampled_frames = sorted(detections_by_frame.keys())
    results, out_video_path = render_and_save(
        out_dir, frame_cache_dir, sampled_frames, staff_by_frame, fps, stride, total_frames,
    )

    present = [r for r in results if r["staff_present"]]
    print(
        f"\nDone. {len(present)} of {len(results)} sampled frames show the staff member.")
    if present:
        print(f"First seen at t={present[0]['timestamp_sec']}s (frame {present[0]['frame_index']}), "
              f"last seen at t={present[-1]['timestamp_sec']}s (frame {present[-1]['frame_index']}).")
    print(f"Annotated video: {out_video_path}")
    print(f"Per-frame results: {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="sample.mp4")
    parser.add_argument("--ref_dir", default="reference_crops")
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--model", choices=["tiny", "full"], default="full")
    parser.add_argument("--rotations", default="0,180",
                        help="comma-separated degrees, e.g. 0,90,180,270")
    parser.add_argument("--stride", type=int, default=1,
                        help="process every Nth frame (1 = every frame)")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--no_average_refs", action="store_true",
                        help="keep multiple reference photos as separate embeddings "
                        "instead of averaging them into one (see reid.py)")
    parser.add_argument("--confirm_threshold", type=float, default=0.90,
                        help="a frame's best-scoring detection must clear this to count "
                        "toward a confirmed streak. Calibrate against your own model.")
    parser.add_argument("--min_streak", type=int, default=3,
                        help="consecutive sampled frames required to trust a streak")
    args = parser.parse_args()

    rotations = tuple(int(x) for x in args.rotations.split(","))
    run(args.video, args.ref_dir, args.out_dir, model=args.model, rotations=rotations,
        stride=args.stride, conf_thresh=args.conf_thresh, average_refs=not args.no_average_refs,
        confirm_threshold=args.confirm_threshold, min_streak=args.min_streak)
