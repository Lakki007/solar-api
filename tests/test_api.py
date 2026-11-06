import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app
R = dict(plant_id=2, irradiation=.72, module_temperature=48.5, ambient_temperature=31.2, timestamp="2020-06-07T10:00:00")
G = json.loads((Path(__file__).parent/"golden.json").read_text())

@pytest.fixture(scope="module")
def c():
    with TestClient(app) as x: yield x

def test_health(c):
    r = c.get("/health"); assert r.status_code == 200 and r.headers["x-request-id"]
    assert r.json()["fingerprint"] == (Path(__file__).parents[1]/"FINGERPRINT.txt").read_text().strip()

def test_model_info(c): assert {"1", "2"} == set(c.get("/model-info").json()["plants"])

@pytest.mark.parametrize("g", G)
def test_golden_parity(c, g):
    body = {k: v for k, v in g.items() if k != "expected_ac_power"}
    assert c.post("/predict", json=body).json()["expected_ac_power"] == g["expected_ac_power"]

def test_batch_matches_single(c):
    b = [{k: v for k, v in g.items() if k != "expected_ac_power"} for g in G]
    out = c.post("/predict/batch", json={"readings": b}).json()["results"]
    assert [x["expected_ac_power"] for x in out] == [g["expected_ac_power"] for g in G]

def test_more_sun_more_output(c):
    a = c.post("/predict", json={**R, "irradiation": .2}).json()["expected_ac_power"]
    assert c.post("/predict", json=R).json()["expected_ac_power"] > a

def test_dark_is_near_zero(c):
    x = c.post("/predict", json={**R, "irradiation": 0, "timestamp": "2020-06-07T01:00:00", "module_temperature": 22, "ambient_temperature": 24}).json()
    assert x["expected_ac_power"] < 1000

def test_verify_and_tamper(c):
    o = c.post("/predict", json=R).json(); a = {**R, "expected_ac_power": o["expected_ac_power"], "receipt": o["receipt"]}
    assert c.post("/verify", json=a).status_code == 200
    assert c.post("/verify", json={**a, "expected_ac_power": a["expected_ac_power"]+1}).status_code == 409
    assert c.post("/verify", json={**a, "irradiation": .5}).status_code == 409
    assert c.post("/verify", json={**a, "receipt": "0"*64}).status_code == 409

@pytest.mark.parametrize("p", [{"irradiation": -1}, {"irradiation": 9}, {"plant_id": 3}, {"timestamp": "bad"},
                               {"timestamp": "2020-06-07T10:00:00Z"}, {"extra": 1}])
def test_rejects(c, p): assert c.post("/predict", json={**R, **p}).status_code == 422

def test_missing_field(c): assert c.post("/predict", json={"plant_id": 1}).status_code == 422
def test_empty_batch(c): assert c.post("/predict/batch", json={"readings": []}).status_code == 422
def test_range_warning(c):
    x = c.post("/predict", json={**R, "irradiation": 1.4}).json(); assert not x["within_training_range"] and x["warnings"]
def test_night_sun_warning(c): assert c.post("/predict", json={**R, "timestamp": "2020-06-07T02:00:00"}).json()["warnings"]
def test_get_on_predict_405(c): assert c.get("/predict").status_code == 405
