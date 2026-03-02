# ASL Detection Web Application

This repository contains a complete ASL inference system:

- `backend/` FastAPI inference service + model assets
- `frontend/` React (Vite) client

## Backend

### Features

- Loads `backend/weights/asl_classifier.pt` and `backend/weights/labels.json` once at startup
- Uses MediaPipe Hands for landmark extraction
- Preprocessing matches training pipeline:
  - wrist-centered normalization
  - scale normalization
  - bone vectors + joint angles + hand_present feature
- Uses `torch.inference_mode()` and CUDA when available
- Endpoints (versioned only):
  - `GET /api/v1/health`
  - `POST /api/v1/predict/image`
  - `WS /api/v1/ws/predict`
- Per-connection temporal smoothing for websocket predictions
- Confidence thresholding (`nothing` fallback)
- CORS enabled

### Structure

```
backend/
  app/
    api/
      v1/
        endpoints/
          health.py
          predict_image.py
          ws_predict.py
    core/
      config.py
      exceptions.py
      logging.py
      middleware.py
    services/
      model_loader.py
      mediapipe_hands.py
      preprocessing.py
      predictor.py
      smoothing.py
    utils/
      image_io.py
      timing.py
    main.py
    state.py
  weights/
    asl_classifier.pt
    labels.json
    calibration.json
  tests/
```

### Local Run

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Tests

```bash
cd backend
pytest -q
```

## Frontend

### Features

- Image upload mode calling `POST /api/v1/predict/image`
- Live webcam mode streaming frames to `WS /api/v1/ws/predict`
- Mirrored camera display in UI only (frames sent unmirrored)
- Shows:
  - prediction label
  - confidence
  - websocket connection status
  - local landmark overlay (MediaPipe Tasks Vision)

### Local Run

```bash
cd frontend
npm install
npm run dev
```

Frontend defaults:

- HTTP API: `http://127.0.0.1:8080` (set in `.env` if you run backend on 8080)
- WS API: `ws://127.0.0.1:8080`

Override via `.env`:

```env
VITE_API_HTTP_BASE=http://127.0.0.1:8080
VITE_API_WS_BASE=ws://127.0.0.1:8080
VITE_API_PREFIX=/api/v1
```

## Docker (Backend)

Build image from repository root:

```bash
docker build -t asl-backend .
```

Run CPU:

```bash
docker run --rm -p 8000:8000 asl-backend
```

Run GPU:

```bash
docker run --rm --gpus all -p 8000:8000 asl-backend
```

## API Details

### `GET /api/v1/health`
Response:
```json
{
  "ok": true,
  "device": "cpu",
  "model_loaded": true
}
```

### `POST /api/v1/predict/image`
Multipart form field: `image` (file).
Response:
```json
{
  "pred": "A",
  "confidence": 0.92,
  "hand_detected": true
}
```

### `WS /api/v1/ws/predict`
Send:
```json
{ "frame": "data:image/jpeg;base64,..." }
```

Optional control message:
```json
{
  "control": {
    "smoothing_window": 15,
    "confidence_threshold": 0.55,
    "send_landmarks": false
  }
}
```

Receive:
```json
{
  "pred": "B",
  "confidence": 0.88,
  "hand_detected": true,
  "landmarks": null
}
```

## Generated Weights Bundle

`backend/weights/` contains:

- `asl_classifier.pt`
- `labels.json`
- `calibration.json`
