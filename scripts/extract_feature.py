import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import mediapipe as mp
try:
    from mediapipe.python.solutions import hands as mp_hands_module
except Exception:
    mp_hands_module = None
try:
    import mediapipe.solutions as mp_solutions_module
except Exception:
    mp_solutions_module = None

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


# -----------------------------
# Feature engineering (robust)
# -----------------------------
# MediaPipe Hands landmark indices:
# 0 wrist
# Thumb: 1,2,3,4
# Index: 5,6,7,8
# Middle: 9,10,11,12
# Ring: 13,14,15,16
# Pinky: 17,18,19,20

# Bone edges (parent -> child)
BONE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
]

# Joint angle triplets (a,b,c) = angle at b formed by (a-b) and (c-b)
ANGLE_TRIPLETS = [
    # thumb
    (1, 2, 3),
    (2, 3, 4),
    # index
    (5, 6, 7),
    (6, 7, 8),
    # middle
    (9, 10, 11),
    (10, 11, 12),
    # ring
    (13, 14, 15),
    (14, 15, 16),
    # pinky
    (17, 18, 19),
    (18, 19, 20),
    # knuckle spreads (between fingers) around palm
    (5, 0, 9),
    (9, 0, 13),
    (13, 0, 17),
]


@dataclass
class PrepConfig:
    image_size_note: str = "original"
    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    feature_version: str = "v1_normxyz_bones_angles_handpresent"
    include_z: bool = True
    include_bones: bool = True
    include_angles: bool = True
    include_hand_present: bool = True


def list_images(root: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(root, f"**/*{ext}"), recursive=True))
    return sorted(paths)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    # u,v: (3,)
    nu = np.linalg.norm(u) + 1e-8
    nv = np.linalg.norm(v) + 1e-8
    cos = float(np.dot(u, v) / (nu * nv))
    cos = max(-1.0, min(1.0, cos))
    return float(np.arccos(cos))  # radians


def normalize_landmarks(pts: np.ndarray) -> np.ndarray:
    """
    pts: (21,3) in MediaPipe normalized coords (x,y,z).
    Normalize:
      - center by wrist
      - scale by max XY distance from wrist
    """
    wrist = pts[0].copy()
    centered = pts - wrist

    # scale by max radius in XY plane for scale invariance
    scale = np.max(np.linalg.norm(centered[:, :2], axis=1)) + 1e-6
    scaled = centered / scale
    return scaled.astype(np.float32)


def build_features(
    pts_norm: np.ndarray, cfg: PrepConfig, hand_present: int
) -> np.ndarray:
    feats = []

    # 1) normalized landmarks
    if cfg.include_z:
        feats.append(pts_norm.reshape(-1))  # 63
    else:
        feats.append(pts_norm[:, :2].reshape(-1))  # 42

    # 2) bone vectors
    if cfg.include_bones:
        bones = []
        for a, b in BONE_EDGES:
            bones.append(pts_norm[b] - pts_norm[a])  # (3,)
        bones = np.stack(bones, axis=0).reshape(-1).astype(np.float32)  # ~20*3=60
        feats.append(bones)

    # 3) angles
    if cfg.include_angles:
        angles = []
        for a, b, c in ANGLE_TRIPLETS:
            u = pts_norm[a] - pts_norm[b]
            v = pts_norm[c] - pts_norm[b]
            angles.append(angle_between(u, v))
        angles = np.array(angles, dtype=np.float32)  # ~13
        feats.append(angles)

    # 4) hand_present
    if cfg.include_hand_present:
        feats.append(np.array([float(hand_present)], dtype=np.float32))

    return np.concatenate(feats, axis=0).astype(np.float32)


def extract_one(
    hands, img_path: str, cfg: PrepConfig
) -> Tuple[Optional[np.ndarray], int]:
    """
    Returns (features, hand_present)
    - If no hand detected: (None, 0) for non-'nothing' classes.
    - For 'nothing' class you can still store a vector with hand_present=0 later.
    """
    img = Image.open(img_path).convert("RGB")
    rgb = np.array(img, dtype=np.uint8)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, 0

    lm = res.multi_hand_landmarks[0].landmark
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)  # (21,3)
    pts_norm = normalize_landmarks(pts)
    feats = build_features(pts_norm, cfg, hand_present=1)
    return feats, 1


def prepare_split(
    split_dir: str,
    labels: List[str],
    label_to_idx: Dict[str, int],
    out_dir: str,
    cfg: PrepConfig,
):
    # Compatibility across MediaPipe package variants:
    # some builds do not expose `mp.solutions` at top-level.
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        hands_ctor = mp.solutions.hands.Hands
    elif mp_solutions_module is not None and hasattr(mp_solutions_module, "hands"):
        hands_ctor = mp_solutions_module.hands.Hands
    elif mp_hands_module is not None and hasattr(mp_hands_module, "Hands"):
        hands_ctor = mp_hands_module.Hands
    else:
        raise RuntimeError(
            "MediaPipe Hands is unavailable. This usually means an incompatible or unofficial "
            "`mediapipe` install. Use Python 3.10/3.11 and reinstall `mediapipe`."
        )

    hands = hands_ctor(
        static_image_mode=True,
        max_num_hands=cfg.max_num_hands,
        min_detection_confidence=cfg.min_detection_confidence,
        min_tracking_confidence=0.5,
    )

    X_list = []
    y_list = []
    stats = {
        "total_images": 0,
        "detected_images": 0,
        "per_class": {lbl: {"total": 0, "detected": 0} for lbl in labels},
        "dropped_non_nothing_nohand": 0,
        "kept_nothing_nohand": 0,
    }

    # Determine feature dimension once from a dummy detected sample
    feature_dim = None

    for lbl in labels:
        class_dir = os.path.join(split_dir, lbl)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        paths = [p for p in list_images(class_dir)]
        stats["per_class"][lbl]["total"] = len(paths)
        stats["total_images"] += len(paths)

        for p in tqdm(paths, desc=f"{os.path.basename(split_dir)}::{lbl}", leave=False):
            feats, hand_present = extract_one(hands, p, cfg)

            if feats is None:
                # No hand detected
                if lbl.lower() == "nothing":
                    # For nothing: keep a valid "no hand" sample as zeros + hand_present=0 (if enabled)
                    # Build a zero vector matching the feature dim:
                    if feature_dim is None:
                        # we don't know dim yet; defer by skipping for now
                        stats["kept_nothing_nohand"] += 1
                        continue
                    zero = np.zeros((feature_dim,), dtype=np.float32)
                    if cfg.include_hand_present:
                        zero[-1] = 0.0
                    X_list.append(zero)
                    y_list.append(label_to_idx[lbl])
                    stats["kept_nothing_nohand"] += 1
                else:
                    # For letter-like classes: drop to avoid poisoning
                    stats["dropped_non_nothing_nohand"] += 1
                continue

            if feature_dim is None:
                feature_dim = int(feats.shape[0])

            X_list.append(feats)
            y_list.append(label_to_idx[lbl])
            stats["detected_images"] += 1
            stats["per_class"][lbl]["detected"] += 1

    if feature_dim is None:
        raise RuntimeError(
            "No hands detected in any image; cannot infer feature dimension."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"X_{os.path.basename(split_dir)}.npy"), X)
    np.save(os.path.join(out_dir, f"y_{os.path.basename(split_dir)}.npy"), y)

    stats["feature_dim"] = feature_dim
    stats["split_dir"] = split_dir
    stats["out_dir"] = out_dir
    stats["config"] = cfg.__dict__

    with open(
        os.path.join(out_dir, f"report_{os.path.basename(split_dir)}.json"), "w"
    ) as f:
        json.dump(stats, f, indent=2)

    hands.close()
    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--out_dir", type=str, default="prepared")
    parser.add_argument("--min_det_conf", type=float, default=0.5)
    args = parser.parse_args()

    labels = [chr(ord("A") + i) for i in range(26)] + ["del", "nothing", "space"]
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    cfg = PrepConfig(min_detection_confidence=float(args.min_det_conf))

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "labels.json"), "w") as f:
        json.dump(labels, f, indent=2)

    with open(os.path.join(args.out_dir, "preprocess.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    tr = prepare_split(args.train_dir, labels, label_to_idx, args.out_dir, cfg)

    # Print a short summary
    def rate(x, y):
        return 0.0 if y == 0 else (100.0 * x / y)

    print("\n=== Extraction Summary ===")
    print(
        f"Train detected: {tr['detected_images']}/{tr['total_images']} ({rate(tr['detected_images'], tr['total_images']):.2f}%)"
    )
    print(f"Feature dim: {tr['feature_dim']}")

    # Per-class detection quick view (train)
    low = []
    for lbl in labels:
        tot = tr["per_class"][lbl]["total"]
        det = tr["per_class"][lbl]["detected"]
        r = rate(det, tot)
        if lbl != "nothing" and r < 95:
            low.append((lbl, r, det, tot))
    if low:
        low = sorted(low, key=lambda x: x[1])
        print("\n⚠️ Classes with <95% detection (train):")
        for lbl, r, det, tot in low[:10]:
            print(f"  {lbl}: {det}/{tot} ({r:.2f}%)")
        print(
            "Consider raising image quality, adjusting min_det_conf, or adding webcam fine-tune data for these."
        )


if __name__ == "__main__":
    main()
