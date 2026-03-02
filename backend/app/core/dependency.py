from __future__ import annotations

from fastapi import Depends
from starlette.requests import HTTPConnection

from app.core.config import Settings
from app.state import AppState


def get_app_state(conn: HTTPConnection) -> AppState:
    return conn.app.state.container


def get_settings(state: AppState = Depends(get_app_state)) -> Settings:
    return state.settings
