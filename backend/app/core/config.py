from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "ASL Detection API"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_json: bool = True
    expose_error_details: bool = False

    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )

    # Absolute defaults so startup works whether uvicorn is launched from project root or backend/.
    weights_dir: Path = BACKEND_DIR / "weights"
    preprocess_config_path: Path = BACKEND_DIR / "weights" / "preprocess.json"
    confidence_threshold: float = 0.55

    max_num_hands: int = 1
    min_detection_confidence: float = 0.3
    min_tracking_confidence: float = 0.5

    ws_smoothing_window: int = 10
    ws_target_fps: float = 30
    max_frame_size: int = 480


@lru_cache
def get_settings() -> Settings:
    return Settings()
