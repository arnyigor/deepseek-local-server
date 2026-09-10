from pathlib import Path

from fastapi.testclient import TestClient

from deepseek_local_server.api.app import create_app
from deepseek_local_server.auth import read_api_token
from deepseek_local_server.config import Settings


class DummyService:
    def runtime_status(self):
        return {"direct_enabled": True, "browser_fallback_enabled": True}

    async def close(self):
        pass



def test_health_is_local_probe_without_auth(tmp_path: Path):
    settings = Settings(home=tmp_path)
    app = create_app(settings, service=DummyService())
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["transport"] == "hybrid"


def test_openai_metadata_requires_local_bearer_token(tmp_path: Path):
    settings = Settings(home=tmp_path)
    app = create_app(settings, service=DummyService())
    token = read_api_token(settings)
    with TestClient(app) as client:
        assert client.get("/v1/models").status_code == 401
        response = client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    ids = {row["id"] for row in response.json()["data"]}
    assert "deepseek-reasoner" in ids
    assert "deepseek-reasoner-search" in ids
