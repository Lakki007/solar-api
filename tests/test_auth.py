from fastapi.testclient import TestClient
from dataclasses import replace
import app.main as m
def test_key(monkeypatch):
    monkeypatch.setattr(m, "settings", replace(m.settings, api_keys=frozenset({"k1"})))
    R = dict(plant_id=2, irradiation=.7, module_temperature=48, ambient_temperature=31, timestamp="2020-06-07T10:00:00")
    with TestClient(m.app) as c:
        assert c.post("/predict", json=R).status_code == 401
        assert c.post("/predict", json=R, headers={"x-api-key": "bad"}).status_code == 401
        assert c.post("/predict", json=R, headers={"x-api-key": "k1"}).status_code == 200
        assert c.get("/health").status_code == 200
