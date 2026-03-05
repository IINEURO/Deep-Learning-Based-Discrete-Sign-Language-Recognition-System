from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

POSE_UPPER_BODY_INDICES: List[int] = [
    0,   # nose
    7, 8,  # ears
    9, 10,  # mouth corners
    11, 12,  # shoulders
    13, 14,  # elbows
    15, 16,  # wrists
    23, 24,  # hips
]

# 21-point hand connections (same for left/right).
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Subset of upper-body pose edges built from MediaPipe's full graph.
POSE_EDGES_ORIG = [
    (7, 3), (8, 6), (7, 9), (8, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
]

POSE_INDEX_TO_LOCAL = {idx: i for i, idx in enumerate(POSE_UPPER_BODY_INDICES)}
POSE_EDGES_LOCAL = [
    (POSE_INDEX_TO_LOCAL[a], POSE_INDEX_TO_LOCAL[b])
    for a, b in POSE_EDGES_ORIG
    if a in POSE_INDEX_TO_LOCAL and b in POSE_INDEX_TO_LOCAL
]


def extract_frame_keypoints(image_bgr: np.ndarray, hands, pose) -> np.ndarray:
    """Extract [55,4] keypoints from one frame.

    Layout:
    - 0:21   left hand
    - 21:42  right hand
    - 42:55  upper-body pose (13 points)
    Each point = [x, y, z, vis]
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid frame for keypoint extraction")

    frame_kpts = np.zeros((55, 4), dtype=np.float32)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    hands_res = hands.process(image_rgb)
    if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
        for hand_lms, handedness in zip(
            hands_res.multi_hand_landmarks, hands_res.multi_handedness
        ):
            label = handedness.classification[0].label.lower()
            offset = 0 if label == "left" else 21
            for i, lm in enumerate(hand_lms.landmark):
                frame_kpts[offset + i] = [lm.x, lm.y, lm.z, 1.0]

    pose_res = pose.process(image_rgb)
    if pose_res.pose_landmarks:
        for local_i, pose_i in enumerate(POSE_UPPER_BODY_INDICES):
            lm = pose_res.pose_landmarks.landmark[pose_i]
            frame_kpts[42 + local_i] = [lm.x, lm.y, lm.z, float(lm.visibility)]

    return frame_kpts


def draw_keypoints(
    frame_bgr: np.ndarray,
    keypoints_55x4: np.ndarray,
    min_vis: float = 0.05,
) -> np.ndarray:
    if keypoints_55x4.shape != (55, 4):
        raise ValueError(f"Expected (55,4), got {keypoints_55x4.shape}")

    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    def to_xy(pt):
        x = int(np.clip(pt[0], 0.0, 1.0) * (w - 1))
        y = int(np.clip(pt[1], 0.0, 1.0) * (h - 1))
        return x, y

    def draw_edges(base: int, edges: List[Tuple[int, int]], color: Tuple[int, int, int]):
        for a, b in edges:
            pa = keypoints_55x4[base + a]
            pb = keypoints_55x4[base + b]
            if pa[3] >= min_vis and pb[3] >= min_vis:
                cv2.line(canvas, to_xy(pa), to_xy(pb), color, 2, cv2.LINE_AA)

    def draw_points(indices: List[int], color: Tuple[int, int, int]):
        for i in indices:
            p = keypoints_55x4[i]
            if p[3] >= min_vis:
                cv2.circle(canvas, to_xy(p), 3, color, -1, cv2.LINE_AA)

    draw_edges(0, HAND_EDGES, (255, 128, 0))
    draw_edges(21, HAND_EDGES, (0, 200, 255))

    for a, b in POSE_EDGES_LOCAL:
        pa = keypoints_55x4[42 + a]
        pb = keypoints_55x4[42 + b]
        if pa[3] >= min_vis and pb[3] >= min_vis:
            cv2.line(canvas, to_xy(pa), to_xy(pb), (0, 255, 0), 2, cv2.LINE_AA)

    draw_points(list(range(0, 21)), (255, 128, 0))
    draw_points(list(range(21, 42)), (0, 200, 255))
    draw_points(list(range(42, 55)), (0, 255, 0))

    return canvas


def load_frames_from_source(source_path: str | Path) -> List[np.ndarray]:
    """Load all frames from a sample dir (images) or a video file."""
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Source path not found: {path}")

    if path.is_dir():
        image_files = sorted(
            [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        )
        if not image_files:
            raise RuntimeError(f"No image frames found in dir: {path}")
        frames = []
        for p in image_files:
            img = cv2.imread(str(p))
            if img is None:
                raise RuntimeError(f"Failed to read image: {p}")
            frames.append(img)
        return frames

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {path}")

    frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return frames
