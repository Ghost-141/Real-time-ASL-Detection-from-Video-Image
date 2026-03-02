import os
import warnings
import json
from collections import deque

import cv2
import numpy as np
import torch
import mediapipe as mp


# -----------------------------
# Must match training pipeline
# -----------------------------
BONE_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]
ANGLE_TRIPLETS = [
    (1, 2, 3),
    (2, 3, 4),
    (5, 6, 7),
    (6, 7, 8),
    (9, 10, 11),
    (10, 11, 12),
    (13, 14, 15),
    (14, 15, 16),
    (17, 18, 19),
    (18, 19, 20),
    (5, 0, 9),
    (9, 0, 13),
    (13, 0, 17),
]

warnings.filterwarnings("ignore", module="mediapipe")


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
    include_z=True,
    include_bones=True,
    include_angles=True,
    include_hand_present=True,
    hand_present=1,
) -> np.ndarray:
    feats = []
    feats.append(pts_norm.reshape(-1) if include_z else pts_norm[:, :2].reshape(-1))

    if include_bones:
        bones = []
        for a, b in BONE_EDGES:
            bones.append(pts_norm[b] - pts_norm[a])
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


def load_model(weights_dir: str, device: str):
    w_path = os.path.join(weights_dir, "asl_classifier.pt")
    if not os.path.exists(w_path):
        raise FileNotFoundError(f"Missing weights: {w_path}")

    model = torch.jit.load(w_path, map_location=device)
    model.eval().to(device)
    return model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--smooth", type=int, default=20)
    parser.add_argument("--conf_thr", type=float, default=0.2)
    parser.add_argument("--min_det_conf", type=float, default=0.2)
    args = parser.parse_args()

    # Load labels
    labels_path = os.path.join(args.weights_dir, "labels.json")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Missing labels.json: {labels_path}")

    with open(labels_path, "r") as f:
        labels = json.load(f)

    # Read preprocess flags if you have them (recommended)
    preprocess_path = os.path.join("prepared", "preprocess.json")
    if os.path.exists(preprocess_path):
        with open(preprocess_path, "r") as f:
            prep = json.load(f)
        include_z = bool(prep.get("include_z", True))
        include_bones = bool(prep.get("include_bones", True))
        include_angles = bool(prep.get("include_angles", True))
        include_hand_present = bool(prep.get("include_hand_present", True))
    else:
        include_z, include_bones, include_angles, include_hand_present = (
            True,
            True,
            True,
            True,
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}  |  classes={len(labels)}")

    # Confidence threshold: use calibration.json if not provided
    conf_thr = args.conf_thr
    calib_path = os.path.join(args.weights_dir, "calibration.json")
    if conf_thr is None and os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            conf_thr = float(json.load(f).get("suggested_conf_threshold", 0.55))
    if conf_thr is None:
        conf_thr = 0.55

    model = load_model(args.weights_dir, device)

    # MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_det_conf,
        min_tracking_confidence=0.5,
    )
    drawer = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    votes = deque(maxlen=int(args.smooth))
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = hands.process(rgb)

        pred = "nothing"
        conf = 0.0

        if res.multi_hand_landmarks:
            # Extract landmarks
            lm = res.multi_hand_landmarks[0].landmark
            pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
            pts_norm = normalize_landmarks(pts)

            # Build features
            feat = build_features(
                pts_norm,
                include_z=include_z,
                include_bones=include_bones,
                include_angles=include_angles,
                include_hand_present=include_hand_present,
                hand_present=1,
            )

            xt = torch.from_numpy(feat).unsqueeze(0).to(device)

            with torch.inference_mode():
                logits = model(xt)
                probs = torch.softmax(logits, dim=1)[0]
                conf_t, idx = torch.max(probs, dim=0)
                conf = float(conf_t.item())
                pred = labels[int(idx)]

            # Draw landmarks
            drawer.draw_landmarks(
                frame,
                res.multi_hand_landmarks[0],
                mp_hands.HAND_CONNECTIONS,
                styles.get_default_hand_landmarks_style(),
                styles.get_default_hand_connections_style(),
            )

        # Production rule: no hand or low confidence -> nothing
        if (not res.multi_hand_landmarks) or (conf < conf_thr):
            pred_use = "nothing"
        else:
            pred_use = pred

        votes.append(pred_use)
        final = max(set(votes), key=votes.count) if votes else pred_use

        frame_display = cv2.flip(frame, 1)

        # Overlay
        class_text = f"{final}"
        base_x = 160
        font = cv2.FONT_HERSHEY_SIMPLEX
        pulse = 0.5 + 0.5 * np.sin(frame_idx * 0.35)
        label_scale = 2.1 + 0.35 * pulse if final != "nothing" else 1.8
        label_thickness = 4 + (1 if pulse > 0.65 and final != "nothing" else 0)

        cv2.putText(
            frame_display,
            class_text,
            (base_x, 130),
            font,
            label_scale,
            (80, 220, 80),
            label_thickness,
            cv2.LINE_AA,
        )

        cv2.imshow("ASL Realtime", frame_display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("]"):
            conf_thr = min(0.99, conf_thr + 0.02)
        if key == ord("["):
            conf_thr = max(0.05, conf_thr - 0.02)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
