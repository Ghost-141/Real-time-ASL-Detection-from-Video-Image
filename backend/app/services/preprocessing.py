from __future__ import annotations

import numpy as np

BONE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

ANGLE_TRIPLETS = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
    (5, 0, 9), (9, 0, 13), (13, 0, 17),
]


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    nu = np.linalg.norm(u) + 1e-8
    nv = np.linalg.norm(v) + 1e-8
    cos = float(np.dot(u, v) / (nu * nv))
    cos = max(-1.0, min(1.0, cos))
    return float(np.arccos(cos))


def normalize_landmarks(pts: np.ndarray) -> np.ndarray:
    wrist = pts[0].copy()
    centered = pts - wrist
    scale = np.max(np.linalg.norm(centered[:, :2], axis=1)) + 1e-6
    return (centered / scale).astype(np.float32)


def build_features(
    pts_norm: np.ndarray,
    include_z: bool = True,
    include_bones: bool = True,
    include_angles: bool = True,
    include_hand_present: bool = True,
    hand_present: int = 1,
) -> np.ndarray:
    feats: list[np.ndarray] = []

    feats.append(pts_norm.reshape(-1) if include_z else pts_norm[:, :2].reshape(-1))

    if include_bones:
        bones = [pts_norm[b] - pts_norm[a] for a, b in BONE_EDGES]
        feats.append(np.stack(bones, axis=0).reshape(-1).astype(np.float32))

    if include_angles:
        angles = []
        for a, b, c in ANGLE_TRIPLETS:
            u = pts_norm[a] - pts_norm[b]
            v = pts_norm[c] - pts_norm[b]
            angles.append(angle_between(u, v))
        feats.append(np.array(angles, dtype=np.float32))

    if include_hand_present:
        feats.append(np.array([float(hand_present)], dtype=np.float32))

    return np.concatenate(feats, axis=0).astype(np.float32)
