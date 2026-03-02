import time
import logging

from fastapi import APIRouter, Depends, File, UploadFile

from app.core.dependency import get_app_state
from app.schemas.predict import PredictResponse
from app.state import AppState
from app.utils.image_io import bytes_to_rgb, resize_rgb

router = APIRouter()
logger = logging.getLogger("app.http")


@router.post('/predict/image', response_model=PredictResponse)
async def predict_image(image: UploadFile = File(...), state: AppState = Depends(get_app_state)) -> PredictResponse:
    start = time.perf_counter()
    content = await image.read()
    rgb = bytes_to_rgb(content)
    rgb = resize_rgb(rgb, state.settings.max_frame_size)
    pred, confidence, hand_detected, _ = state.predictor.predict_rgb(rgb)
    duration_ms = round((time.perf_counter() - start) * 1000.0, 2)
    logger.info(
        "POST /api/v1/predict/image -> 200",
        extra={"extras": {"duration_ms": duration_ms}},
    )
    return PredictResponse(pred=pred, confidence=confidence, hand_detected=hand_detected)
