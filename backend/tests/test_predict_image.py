from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _make_blank_jpeg() -> bytes:
    image = Image.new('RGB', (224, 224), color=(0, 0, 0))
    buf = BytesIO()
    image.save(buf, format='JPEG')
    return buf.getvalue()


def test_predict_image_accepts_upload() -> None:
    with TestClient(app) as client:
        files = {'image': ('frame.jpg', _make_blank_jpeg(), 'image/jpeg')}
        response = client.post('/api/v1/predict/image', files=files)

    assert response.status_code == 200
    payload = response.json()
    assert set(payload.keys()) == {'pred', 'confidence', 'hand_detected'}
    assert isinstance(payload['pred'], str)
    assert isinstance(payload['confidence'], float)
    assert isinstance(payload['hand_detected'], bool)
