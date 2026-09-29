"""
detector.py
-----------
Wraps a Darknet YOLO model (loaded through OpenCV's DNN module) to detect
people ("person" class only) in a single frame.
"""

import cv2
import numpy as np


class PersonDetector:
    def __init__(self, cfg_path, weights_path, classes_path, input_size=416):
        # cv2.dnn.readNetFromDarknet loads the network architecture (cfg)
        # and its trained weights. This is the same file format the
        # original Darknet/YOLO authors ship, so any YOLOv3/v4-family
        # .cfg + .weights pair from the official repos works here.
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        # YOLO's Darknet-style output layers are the last layer of each
        # detection "head" (there are 2 for -tiny models, 3 for full
        # models). getUnconnectedOutLayersNames() finds them automatically
        # so we don't have to hardcode layer names per model variant.
        self.output_layers = self.net.getUnconnectedOutLayersNames()

        with open(classes_path) as f:
            self.classes = [line.strip() for line in f]
        self.person_idx = self.classes.index("person")

        self.input_size = input_size

    def _detect_single_orientation(self, frame, conf_thresh):
        """Run the network once, on the frame exactly as given (no rotation)."""
        h, w = frame.shape[:2]

        # blobFromImage does 3 things: scales pixel values from [0,255] to
        # [0,1] (the 1/255.0 factor), resizes to the network's expected
        # square input (self.input_size), and swaps BGR->RGB (OpenCV loads
        # images as BGR, Darknet models expect RGB).
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (self.input_size, self.input_size),
            swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes, confidences = [], []
        for output in outputs:
            # Each row in `output` is one candidate detection:
            # [center_x, center_y, width, height, objectness, class_0_score, class_1_score, ...]
            # all normalized to [0,1] relative to the input frame size.
            for det in output:
                class_scores = det[5:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                if class_id == self.person_idx and confidence > conf_thresh:
                    cx, cy = det[0] * w, det[1] * h
                    bw, bh = det[2] * w, det[3] * h
                    x, y = int(cx - bw / 2), int(cy - bh / 2)
                    boxes.append([x, y, int(bw), int(bh)])
                    confidences.append(float(confidence))
        return boxes, confidences

    @staticmethod
    def _rotate_box_to_original(box, k, rotated_w, rotated_h):
        """
        Map a box detected on a rotated frame back to the original frame's
        coordinate system. `k` = number of 90-degree counter-clockwise
        rotations np.rot90 applied (1, 2, or 3). rotated_w/rotated_h are the
        width/height of the frame that was actually fed to the detector
        (post-rotation).
        """
        x, y, w, h = box
        for _ in range(k):
            # Undo one 90-degree counter-clockwise rotation. If np.rot90
            # took a (H, W) frame to a (W, H) frame, a box at (x, y, w, h)
            # in the rotated frame came from (y, W_before - x - w, h, w)
            # in the frame one rotation step earlier.
            x, y, w, h, rotated_w, rotated_h = (
                y,
                rotated_w - x - w,
                h,
                w,
                rotated_h,
                rotated_w,
            )
        return [x, y, w, h]

    def detect(self, frame, conf_thresh=0.25, nms_thresh=0.45, rotations=(0,)):
        """
        Detect people in `frame`, optionally merging results from multiple
        rotated copies of the frame (see module docstring for why).

        rotations: tuple of degrees to try, drawn from {0, 90, 180, 270}.
        e.g. rotations=(0, 180) covers the two most common orientations in
        a top-down ceiling shot for ~2x the compute of rotations=(0,).
        """
        all_boxes, all_confs = [], []
        for deg in rotations:
            k = (deg // 90) % 4  # how many 90-degree CCW rotations
            rotated = np.rot90(frame, k) if k else frame
            rh, rw = rotated.shape[:2]
            boxes, confs = self._detect_single_orientation(
                rotated, conf_thresh)
            for box in boxes:
                all_boxes.append(self._rotate_box_to_original(box, k, rw, rh))
            all_confs.extend(confs)

        if not all_boxes:
            return []

        # Because the same person can be picked up by more than one
        # rotation pass, we run NMS again across the merged set to collapse
        # duplicate/overlapping boxes into one.
        idxs = cv2.dnn.NMSBoxes(all_boxes, all_confs, conf_thresh, nms_thresh)
        idxs = idxs.flatten() if len(idxs) else []
        return [
            {"bbox": all_boxes[i], "conf": all_confs[i]} for i in idxs
        ]
