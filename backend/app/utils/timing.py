from __future__ import annotations

import time


class FrameGate:
    def __init__(self, target_fps: float) -> None:
        fps = max(1.0, float(target_fps))
        self._interval = 1.0 / fps
        self._next = 0.0

    def allow_now(self) -> bool:
        now = time.monotonic()
        if now < self._next:
            return False
        self._next = now + self._interval
        return True
