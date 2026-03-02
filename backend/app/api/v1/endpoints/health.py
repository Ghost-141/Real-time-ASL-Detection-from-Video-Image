from fastapi import APIRouter, Depends

from app.core.dependency import get_app_state
from app.schemas.health import HealthResponse
from app.state import AppState

router = APIRouter()


@router.get('/health', response_model=HealthResponse)
def health(state: AppState = Depends(get_app_state)) -> HealthResponse:
    return HealthResponse(ok=True, device=state.predictor.device, model_loaded=state.predictor.model is not None)
