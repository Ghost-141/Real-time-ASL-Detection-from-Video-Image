from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.services.mediapipe_hands import HandsService
from app.services.model_loader import load_model_bundle
from app.services.predictor import Predictor
from app.state import AppState
import numpy as np


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.debug, settings.log_level, settings.log_json)

    model_bundle = load_model_bundle(settings.weights_dir)
    hands = HandsService(
        max_num_hands=settings.max_num_hands,
        min_detection_confidence=settings.min_detection_confidence,
        min_tracking_confidence=settings.min_tracking_confidence,
    )
    predictor = Predictor(
        model=model_bundle.model,
        labels=model_bundle.labels,
        device=model_bundle.device,
        confidence_threshold=model_bundle.calibration_threshold or settings.confidence_threshold,
        preprocess_config_path=settings.preprocess_config_path,
        hands_service=hands,
    )

    # Warmup: run a tiny dummy frame through MediaPipe + model once at startup.
    dummy = np.zeros((settings.max_frame_size, settings.max_frame_size, 3), dtype=np.uint8)
    predictor.predict_rgb(dummy)

    app.state.container = AppState(settings=settings, predictor=predictor, hands=hands)
    yield
    hands.close()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

    register_exception_handlers(app)
    # Expose only versioned routes.
    # Serve model assets for frontend landmark detection.
    app.mount("/weights", StaticFiles(directory=str(settings.weights_dir)), name="weights")

    if settings.api_v1_prefix:
        app.include_router(api_router, prefix=settings.api_v1_prefix)
    else:
        app.include_router(api_router)
    return app


app = create_app()
