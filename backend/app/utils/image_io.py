from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from app.core.exceptions import InvalidImageError


def bytes_to_rgb(raw: bytes) -> np.ndarray:
    if not raw:
        raise InvalidImageError('Empty image payload')
    try:
        image = Image.open(BytesIO(raw)).convert('RGB')
    except Exception as exc:
        raise InvalidImageError(f'Unable to decode image: {exc}') from exc
    return np.array(image, dtype=np.uint8)


def base64_to_rgb(data: str) -> np.ndarray:
    payload = data.split(',', 1)[1] if ',' in data else data
    try:
        raw = base64.b64decode(payload)
    except Exception as exc:
        raise InvalidImageError(f'Invalid base64 frame: {exc}') from exc
    return bytes_to_rgb(raw)


def resize_rgb(rgb: np.ndarray, max_size: int | None) -> np.ndarray:
    if max_size is None or max_size <= 0:
        return rgb
    h, w = rgb.shape[:2]
    if max(h, w) <= max_size:
        return rgb
    scale = float(max_size) / float(max(h, w))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def rgb_to_jpeg_bytes(rgb: np.ndarray, quality: int = 90) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise InvalidImageError('JPEG encoding failed')
    return encoded.tobytes()
