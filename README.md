# ASL Detection

A real-time American Sign Language (ASL) recognition system built with a **FastAPI backend** and a **React + Vite frontend**. The backend serves a **serialized PyTorch model** for efficient production inference and uses **MediaPipe hand landmarks** for robust hand feature extraction.

The system supports:

* **Single image upload prediction**
* **Live webcam inference via WebSocket streaming**

Designed for low-latency, scalable deployment, this application enables real-time ASL sign recognition directly from images or video streams, making it suitable for accessibility tools, educational platforms, and assistive communication systems.


## Table of Contents
- [ASL Detection](#asl-detection)
  - [Table of Contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Current Project Structure](#current-project-structure)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
    - [Backend](#backend)
    - [Frontend](#frontend)
  - [Configuration](#configuration)
    - [Backend variables](#backend-variables)
    - [Frontend variables](#frontend-variables)
  - [API Documentation](#api-documentation)
    - [Health](#health)
    - [Image prediction](#image-prediction)
    - [WebSocket live prediction](#websocket-live-prediction)
    - [HTTP error shape](#http-error-shape)
  - [How the System Works](#how-the-system-works)
  - [Testing](#testing)
  - [Training](#training)
  - [Deployment Notes](#deployment-notes)
    - [Docker](#docker)

## Architecture

- Backend: FastAPI app with startup-loaded model, shared app state, JSON logging, and custom exception handlers.
- Inference path: image/frame decode -> resize -> MediaPipe hand detection -> feature extraction -> MLP classification -> confidence gating.
- Live mode stability: WebSocket frame throttling (`FrameGate`) + majority-vote smoothing window.
- Frontend:
  - Upload mode calls `POST /predict/image`.
  - Live mode streams webcam frames over WebSocket (`/ws/predict`) and applies server-side smoothing.
  - Optional browser-side landmark overlay uses `@mediapipe/tasks-vision` and model asset served from backend `/weights/hand_landmarker.task`.

## Current Project Structure

```markdown
.
├─ frontend
├─ backend/
│  ├─ app/
│  │  ├─ __init__.py
│  │  ├─ main.py
│  │  ├─ state.py
│  │  ├─ api/
│  │  │  ├─ __init__.py
│  │  │  └─ v1/
│  │  │     ├─ __init__.py
│  │  │     ├─ router.py
│  │  │     └─ endpoints/
│  │  │        ├─ __init__.py
│  │  │        ├─ health.py
│  │  │        ├─ predict_image.py
│  │  │        └─ ws_predict.py
│  │  ├─ core/
│  │  │  ├─ __init__.py
│  │  │  ├─ config.py
│  │  │  ├─ dependency.py
│  │  │  ├─ exceptions.py
│  │  │  └─ logging.py
│  │  ├─ schemas/
│  │  │  ├─ __init__.py
│  │  │  ├─ health.py
│  │  │  └─ predict.py
│  │  ├─ services/
│  │  │  ├─ __init__.py
│  │  │  ├─ mediapipe_hands.py
│  │  │  ├─ model_loader.py
│  │  │  ├─ predictor.py
│  │  │  ├─ preprocessing.py
│  │  │  └─ smoothing.py
│  │  └─ utils/
│  │     ├─ __init__.py
│  │     ├─ image_io.py
│  │     └─ timing.py
│  ├─ tests/
│  │  ├─ __init__.py
│  │  ├─ test_health.py
│  │  ├─ test_predict_image.py
│  ├─ weights/
│  │  ├─ asl_classifier.pt
│  │  ├─ labels.json
│  │  ├─ preprocess.json
│  │  ├─ calibration.json
│  │  └─ hand_landmarker.task
│  └─ .env.example
├─ scripts/
│  ├─ weights/
│  ├─ prepared/
│  ├─ train.py
│  ├─ extract_features.py
│  └─ prediction.py
├─ dataset/
│  └─ (raw/ processed/ splits/ etc.)
├─ Dockerfile
├─ .gitignore
├─ README.md
└─ pyproject.toml
```

## Prerequisites

- Python `3.11+`
- Node.js `18+` (Node `20` recommended)
- Webcam for live mode
- Optional: NVIDIA GPU (CUDA) for lower model latency

## Local Setup

### Backend

```bash
cd backend
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate
pip install uv
uv sync 
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Backend base URL: `http://127.0.0.1:8080`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend URL: `http://127.0.0.1:5173`

## Configuration

The backend reads environment variables via `pydantic-settings` (`.env` supported in `backend/`).

### Backend variables

| Variable                   | Default                                             | Notes                                                      |
| -------------------------- | --------------------------------------------------- | ---------------------------------------------------------- |
| `APP_NAME`                 | `ASL Detection API`                                 | FastAPI app title                                          |
| `DEBUG`                    | `false`                                             | Enables debug logging style                                |
| `API_V1_PREFIX`            | `/api/v1`                                           | Route prefix                                               |
| `LOG_LEVEL`                | `INFO`                                              | Standard Python logging levels                             |
| `LOG_JSON`                 | `true`                                              | Structured JSON logs when not in debug                     |
| `EXPOSE_ERROR_DETAILS`     | `false`                                             | Include internal error details in API responses            |
| `CORS_ORIGINS`             | `["http://localhost:5173","http://127.0.0.1:5173"]` | Allowed frontend origins                                   |
| `WEIGHTS_DIR`              | `backend/weights`                                   | Contains model + labels + calibration + hand landmark task |
| `PREPROCESS_CONFIG_PATH`   | `backend/weights/preprocess.json`                   | Feature flags used by predictor                            |
| `CONFIDENCE_THRESHOLD`     | `0.55`                                              | Fallback threshold if calibration file missing             |
| `MAX_NUM_HANDS`            | `1`                                                 | MediaPipe hand count                                       |
| `MIN_DETECTION_CONFIDENCE` | `0.3`                                               | MediaPipe detect threshold                                 |
| `MIN_TRACKING_CONFIDENCE`  | `0.5`                                               | MediaPipe track threshold                                  |
| `WS_SMOOTHING_WINDOW`      | `10`                                                | Default WS majority vote window                            |
| `WS_TARGET_FPS`            | `30`                                                | Frame admission rate limiter                               |
| `MAX_FRAME_SIZE`           | `480`                                               | Max image/frame side length before inference               |

### Frontend variables

Set in `frontend/.env` (or `.env.local`):

```env
VITE_API_HTTP_BASE=http://127.0.0.1:8080
VITE_API_WS_BASE=ws://127.0.0.1:8080
VITE_API_PREFIX=/api/v1
```

## API Documentation

OpenAPI docs are available at:
- `http://127.0.0.1:8080/docs`
- `http://127.0.0.1:8080/redoc`

If `API_V1_PREFIX=/api/v1`, endpoint paths below are prefixed with `/api/v1`.

### Health

`GET /health`

Response:
```json
{
  "ok": true,
  "device": "cpu",
  "model_loaded": true
}
```

### Image prediction

`POST /predict/image`  
Content-Type: `multipart/form-data`  
Field name: `image`

Example:
```bash
curl -X POST "http://127.0.0.1:8080/api/v1/predict/image" \
  -F "image=@sample.jpg"
```

Response:
```json
{
  "pred": "A",
  "confidence": 0.92,
  "hand_detected": true
}
```

Notes:
- Returns `pred: "nothing"` when no hand is found or confidence is below threshold.
- `hand_detected` indicates whether MediaPipe detected a hand before threshold gating.

### WebSocket live prediction

`WS /ws/predict`

Client can send either:
- frame payload:
```json
{
  "frame": "data:image/jpeg;base64,..."
}
```
- or image alias:
```json
{
  "image": "data:image/jpeg;base64,..."
}
```
- and/or control-only updates:
```json
{
  "control": {
    "smoothing_window": 15,
    "confidence_threshold": 0.55,
    "send_landmarks": false
  }
}
```

Control ranges:
- `smoothing_window`: `1..60`
- `confidence_threshold`: `0.05..0.99`

Server response:
```json
{
  "pred": "B",
  "confidence": 0.88,
  "hand_detected": true,
  "landmarks": null
}
```

Error responses can include:
```json
{ "detail": "Invalid JSON payload" }
```

### HTTP error shape

Custom exception handlers return:
```json
{
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Image could not be decoded or validated.",
    "request_id": null
  }
}
```

## How the System Works

1. Backend startup loads model assets from `backend/weights`, initializes MediaPipe Hands, and performs a warmup inference on a dummy frame.
2. Input ingestion:
   - Upload mode: multipart image bytes.
   - Live mode: base64-encoded JPEG frames over WebSocket.
3. Preprocessing:
   - Decode into RGB.
   - Resize to `MAX_FRAME_SIZE` while preserving aspect ratio.
4. Hand detection:
   - MediaPipe returns 21 landmarks for the first detected hand.
   - If no hand: returns `pred="nothing"`, `confidence=0.0`, `hand_detected=false`.
5. Feature generation:
   - Wrist-centered, scale-normalized landmarks.
   - Optional feature blocks (configured by `preprocess.json`): z-coordinates, bone vectors, joint angles, hand-present flag.
6. Classification:
   - TorchScript model inference (`torch.inference_mode()`).
   - Softmax confidence + argmax class selection.
7. Confidence gating:
   - If confidence < threshold, output is forced to `pred="nothing"` with `hand_detected=true`.
8. Live smoothing:
   - Majority vote over last N predictions for temporal stability.
   - Frame gate limits server processing rate to configured target FPS.

## Testing

From `backend/`:

```bash
pytest -q
```

Current tests cover:
- Health endpoint contract
- Image prediction endpoint contract

## Training

Training utilities are in `scripts/`:
- `extract_feature.py`: dataset feature extraction with MediaPipe
- `train.py`: model training + TorchScript export
- `prediction.py`: local webcam inference script

## Deployment Notes

- Use TLS in production (`wss://` for WebSockets).
- Keep `CORS_ORIGINS` restricted to trusted domains.
- Set `EXPOSE_ERROR_DETAILS=false` in production.
- Ensure the `backend/weights` directory is mounted and immutable at runtime.

### Docker

The docker file only contains the backend setup for deployment. The frontend can be included here based on the deployment stratagey.

```bash
docker build -t asl-detection .

# CPU run
docker run --rm -p 8000:8000 asl-detection

# GPU run (NVIDIA runtime)
docker run --rm --gpus all -p 8000:8000 asl-detection
```
