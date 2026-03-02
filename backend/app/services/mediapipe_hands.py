from __future__ import annotations

import os
import threading

# Silence noisy MediaPipe/absl logs before importing mediapipe.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from absl import logging as absl_logging

absl_logging.set_verbosity(absl_logging.ERROR)

import mediapipe as mp
import numpy as np


class HandsService:
    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.2,
        min_tracking_confidence: float = 0.3,
    ) -> None:
        self._lock = threading.Lock()
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb: np.ndarray):
        with self._lock:
            return self._hands.process(rgb)

    def close(self) -> None:
        with self._lock:
            self._hands.close()
