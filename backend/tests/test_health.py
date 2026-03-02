from fastapi.testclient import TestClient

from app.main import app


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get('/api/v1/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['ok'] is True
    assert payload['model_loaded'] is True
    assert payload['device'] in {'cpu', 'cuda'}
