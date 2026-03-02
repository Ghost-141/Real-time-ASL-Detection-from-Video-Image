from pydantic import BaseModel


class HealthResponse(BaseModel):
    ok: bool
    device: str
    model_loaded: bool
