from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.dependency import get_app_state
from app.schemas.predict import WSFrameIn, WSFrameOut
from app.services.smoothing import MajorityVoteSmoother
from app.state import AppState
from app.utils.image_io import base64_to_rgb, resize_rgb
from app.utils.timing import FrameGate

router = APIRouter()
logger = logging.getLogger("app.ws")


@router.websocket('/ws/predict')
async def ws_predict(websocket: WebSocket, state: AppState = Depends(get_app_state)) -> None:
    await websocket.accept()
    client = websocket.client
    logger.info(
        "ws connected",
        extra={"extras": {"client": f"{client.host}:{client.port}" if client else "unknown"}},
    )
    smoother = MajorityVoteSmoother(maxlen=state.settings.ws_smoothing_window)
    gate = FrameGate(target_fps=state.settings.ws_target_fps)
    send_landmarks = False
    confidence_threshold = None

    try:
        while True:
            data = await websocket.receive_text()
            if not gate.allow_now():
                continue

            try:
                payload = WSFrameIn.model_validate_json(data)
            except ValidationError:
                await websocket.send_json({'detail': 'Invalid JSON payload'})
                continue

            if payload.control:
                if payload.control.smoothing_window:
                    smoother.set_maxlen(int(payload.control.smoothing_window))
                if payload.control.send_landmarks is not None:
                    send_landmarks = bool(payload.control.send_landmarks)
                if payload.control.confidence_threshold is not None:
                    confidence_threshold = float(payload.control.confidence_threshold)

            b64 = payload.frame or payload.image
            if not b64:
                # Allow control-only messages without a frame.
                if payload.control:
                    continue
                await websocket.send_json({'detail': 'Missing frame/image field'})
                continue

            rgb = base64_to_rgb(b64)
            rgb = resize_rgb(rgb, state.settings.max_frame_size)
            pred, confidence, hand_detected, landmarks = state.predictor.predict_rgb(
                rgb,
                return_landmarks=send_landmarks,
                confidence_threshold=confidence_threshold,
            )
            smooth_pred = smoother.push(pred)
            out = WSFrameOut(
                pred=smooth_pred,
                confidence=confidence,
                hand_detected=hand_detected,
                landmarks=landmarks,
            )
            await websocket.send_text(out.model_dump_json())
    except WebSocketDisconnect:
        logger.info(
            "ws disconnected",
            extra={"extras": {"client": f"{client.host}:{client.port}" if client else "unknown"}},
        )
        return
    except Exception as exc:
        logger.exception(
            "ws prediction failure",
            exc_info=exc,
            extra={"extras": {"client": f"{client.host}:{client.port}" if client else "unknown"}},
        )
        await websocket.send_json({'detail': f'Prediction failure: {exc}'})
