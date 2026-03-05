from __future__ import annotations

import numpy as np

# Pose points are appended at keypoint indices 42..54.
LEFT_SHOULDER_IDX = 47
RIGHT_SHOULDER_IDX = 48
LEFT_HIP_IDX = 53
RIGHT_HIP_IDX = 54
POSE_START_IDX = 42
POSE_END_IDX = 55
EPS = 1e-6


def _validate_keypoints(keypoints: np.ndarray) -> None:
    if keypoints.ndim != 3 or keypoints.shape[1:] != (55, 4):
        raise ValueError(
            f"Expected keypoints shape [T,55,4], got {keypoints.shape}"
        )


def normalize_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Per-sample geometric normalization.

    1) Center by shoulder center when available, otherwise pose visible mean.
    2) Scale by shoulder distance; fallback to torso size; fallback to visible spread.
    3) Missing points (vis<=0) stay zero.
    """
    _validate_keypoints(keypoints)

    x = keypoints.astype(np.float32, copy=True)
    coords = x[..., :3]
    vis = x[..., 3] > 0.0  # [T,55]
    t = x.shape[0]

    l_sh = coords[:, LEFT_SHOULDER_IDX, :]
    r_sh = coords[:, RIGHT_SHOULDER_IDX, :]
    l_hip = coords[:, LEFT_HIP_IDX, :]
    r_hip = coords[:, RIGHT_HIP_IDX, :]

    has_l_sh = vis[:, LEFT_SHOULDER_IDX]
    has_r_sh = vis[:, RIGHT_SHOULDER_IDX]
    has_l_hip = vis[:, LEFT_HIP_IDX]
    has_r_hip = vis[:, RIGHT_HIP_IDX]

    # Anchor.
    shoulder_center = 0.5 * (l_sh + r_sh)

    pose_vis = vis[:, POSE_START_IDX:POSE_END_IDX]  # [T,13]
    pose_coords = coords[:, POSE_START_IDX:POSE_END_IDX, :]  # [T,13,3]
    pose_w = pose_vis.astype(np.float32)[..., None]  # [T,13,1]
    pose_count = np.clip(pose_w.sum(axis=1), 1.0, None)  # [T,1]
    pose_center = (pose_coords * pose_w).sum(axis=1) / pose_count  # [T,3]

    anchor = np.zeros((t, 3), dtype=np.float32)
    use_sh_center = has_l_sh & has_r_sh
    anchor[use_sh_center] = shoulder_center[use_sh_center]
    anchor[~use_sh_center] = pose_center[~use_sh_center]

    # Scale.
    sh_dist = np.linalg.norm(l_sh - r_sh, axis=1)  # [T]
    hip_center = 0.5 * (l_hip + r_hip)
    torso_dist = np.linalg.norm(shoulder_center - hip_center, axis=1)  # [T]

    scale = np.ones((t,), dtype=np.float32)
    use_sh_scale = use_sh_center & (sh_dist > EPS)
    scale[use_sh_scale] = sh_dist[use_sh_scale]

    need_alt = ~use_sh_scale
    use_torso = need_alt & has_l_hip & has_r_hip & (torso_dist > EPS)
    scale[use_torso] = torso_dist[use_torso]

    remaining = ~use_sh_scale & ~use_torso
    if np.any(remaining):
        # Fallback to average visible xy spread to anchor.
        xy = coords[..., :2]
        anchor_xy = anchor[:, None, :2]
        dxy = np.linalg.norm(xy - anchor_xy, axis=-1)  # [T,55]
        v = vis.astype(np.float32)
        denom = np.clip(v.sum(axis=1), 1.0, None)
        mean_spread = (dxy * v).sum(axis=1) / denom
        use_spread = remaining & (mean_spread > EPS)
        scale[use_spread] = mean_spread[use_spread]

    scale = np.clip(scale, EPS, None)

    coords_norm = (coords - anchor[:, None, :]) / scale[:, None, None]
    coords_norm[~vis] = 0.0

    out = x
    out[..., :3] = coords_norm
    return out


def compute_velocity(keypoints: np.ndarray) -> np.ndarray:
    """First-order temporal difference on normalized keypoints.

    Returns [T,55,4] where xyz are deltas and vis is valid-pair mask.
    """
    _validate_keypoints(keypoints)

    vel = np.zeros_like(keypoints, dtype=np.float32)
    coords = keypoints[..., :3]
    vis = keypoints[..., 3:4]

    vel[1:, :, :3] = coords[1:] - coords[:-1]
    pair_vis = np.minimum(vis[1:], vis[:-1])
    vel[1:, :, 3:4] = pair_vis

    # Keep missing-pair velocities zero.
    valid = vel[..., 3] > 0.0
    vel[..., :3][~valid] = 0.0
    return vel


def build_sequence_features(
    keypoints: np.ndarray,
    normalize: bool = True,
    use_velocity: bool = True,
    flatten: bool = True,
) -> np.ndarray:
    """Build model input features from raw keypoints.

    Base feature: [T,55,4] (or flattened [T,220]).
    If velocity enabled: concat [T,55,4] velocity -> [T,55,8] (or [T,440]).
    """
    _validate_keypoints(keypoints)

    base = normalize_keypoints(keypoints) if normalize else keypoints.astype(np.float32, copy=True)
    feats = [base]
    if use_velocity:
        feats.append(compute_velocity(base))

    out = np.concatenate(feats, axis=2)
    if flatten:
        t = out.shape[0]
        out = out.reshape(t, -1)
    return out.astype(np.float32, copy=False)
