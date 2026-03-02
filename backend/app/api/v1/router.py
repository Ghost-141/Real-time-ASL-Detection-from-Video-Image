from fastapi import APIRouter

from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.predict_image import router as predict_image_router
from app.api.v1.endpoints.ws_predict import router as ws_predict_router

api_router = APIRouter()
api_router.include_router(health_router, tags=['health'])
api_router.include_router(predict_image_router, tags=['predict'])
api_router.include_router(ws_predict_router, tags=['predict'])
