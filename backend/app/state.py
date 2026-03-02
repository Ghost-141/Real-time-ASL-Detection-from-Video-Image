from dataclasses import dataclass

from app.core.config import Settings
from app.services.mediapipe_hands import HandsService
from app.services.predictor import Predictor


@dataclass
class AppState:
    settings: Settings
    predictor: Predictor
    hands: HandsService
