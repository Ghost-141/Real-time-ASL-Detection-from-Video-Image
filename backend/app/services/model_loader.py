from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ModelBundle:
    model: torch.jit.ScriptModule
    labels: list[str]
    device: str
    calibration_threshold: float | None


def load_model_bundle(weights_dir: Path) -> ModelBundle:
    model_path = weights_dir / 'asl_classifier.pt'
    labels_path = weights_dir / 'labels.json'
    calibration_path = weights_dir / 'calibration.json'

    if not model_path.exists():
        raise FileNotFoundError(f'Missing model file: {model_path}')
    if not labels_path.exists():
        raise FileNotFoundError(f'Missing labels file: {labels_path}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.jit.load(str(model_path), map_location=device)
    model.eval().to(device)

    labels = json.loads(labels_path.read_text(encoding='utf-8'))

    calibration_threshold = None
    if calibration_path.exists():
        payload = json.loads(calibration_path.read_text(encoding='utf-8'))
        calibration_threshold = float(payload.get('suggested_conf_threshold', 0.55))

    return ModelBundle(
        model=model,
        labels=labels,
        device=device,
        calibration_threshold=calibration_threshold,
    )
