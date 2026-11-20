"""Edge cases for inputs: dark, missing/zero temperatures, out-of-range, unit mistakes, bad types."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

DAY = dict(irradiation=.72, module_temperature=48.5, ambient_temperature=31.2, timestamp="2020-06-07T10:00:00")
NIGHT = dict(irradiation=0, module_temperature=22, ambient_temperature=24, timestamp="2020-06-07T01:00:00")
PLANTS = [1, 2]


@pytest.fixture(scope="module")
def c():
    with TestClient(app) as x: yield x


def p(c, body): return c.post("/predict", json=body)


# ---- dark rule: output exactly 0 whatever the temperatures ----
@pytest.mark.parametrize("pid", PLANTS)
@pytest.mark.parametrize("temps", [(22, 24), (0, 0), (None, None), (0, 30), (35, None), (-10, -10), (28, 28)])
@pytest.mark.parametrize("t", ["01:00", "12:00", "19:45"])
def test_dark_is_exactly_zero(c, pid, temps, t):
    x = p(c, {**NIGHT, "plant_id": pid, "module_temperature": temps[0], "ambient_temperature": temps[1],
              "timestamp": f"2020-06-07T{t}:00"}).json()
    assert x["expected_ac_power"] == 0 and x["regime"] == "dark" and x["reliability"] == "ok"


@pytest.mark.parametrize("irr", [0.0, 0.01, 0.02, -0.03])
def test_dark_threshold(c, irr):
    x = p(c, {**NIGHT, "plant_id": 2, "irradiation": irr, "timestamp": "2020-06-07T12:00:00"}).json()
    assert x["expected_ac_power"] == 0 and x["regime"] == "dark"


def test_small_negative_warns(c):
    assert any("negative" in w for w in p(c, {**NIGHT, "plant_id": 1, "irradiation": -0.03}).json()["warnings"])


def test_just_above_dark_uses_model(c):
    assert p(c, {**DAY, "plant_id": 1, "irradiation": 0.05, "timestamp": "2020-06-07T06:30:00",
                 "module_temperature": 25, "ambient_temperature": 24}).json()["regime"] == "daylight"


def test_hot_module_in_dark_warns(c):
    assert any("hotter" in w for w in p(c, {**NIGHT, "plant_id": 1, "module_temperature": 45}).json()["warnings"])


# ---- missing / zero temperatures in daylight ----
@pytest.mark.parametrize("pid", PLANTS)
@pytest.mark.parametrize("field,val", [("module_temperature", None), ("module_temperature", 0),
                                       ("ambient_temperature", None), ("ambient_temperature", 0)])
def test_one_temp_missing_is_inferred(c, pid, field, val):
    full = p(c, {**DAY, "plant_id": pid}).json()
    x = p(c, {**DAY, "plant_id": pid, field: val}).json()
    assert x["reliability"] == "reduced" and any("inferred" in w for w in x["warnings"])
    assert x["inputs_used"][field] not in (None, 0)
    assert abs(x["expected_ac_power"] - full["expected_ac_power"]) < 0.25 * full["expected_ac_power"]


@pytest.mark.parametrize("pid", PLANTS)
@pytest.mark.parametrize("temps", [(None, None), (0, 0), (0, None)])
def test_both_temps_missing(c, pid, temps):
    full = p(c, {**DAY, "plant_id": pid}).json()
    x = p(c, {**DAY, "plant_id": pid, "module_temperature": temps[0], "ambient_temperature": temps[1]}).json()
    assert x["reliability"] == "low" and x["expected_ac_power"] > 0.5 * full["expected_ac_power"]


def test_omitted_temperature_is_422(c):
    body = {**DAY, "plant_id": 1}; del body["ambient_temperature"]
    assert p(c, body).status_code == 422


# ---- outside training range: clamped, never extrapolated ----
@pytest.mark.parametrize("pid", PLANTS)
@pytest.mark.parametrize("override", [{"module_temperature": 95}, {"ambient_temperature": 5},
                                      {"module_temperature": -5, "ambient_temperature": -5}, {"irradiation": 1.45}])
def test_out_of_range_is_clamped(c, pid, override):
    x = p(c, {**DAY, "plant_id": pid, **override}).json()
    assert x["reliability"] == "outside_envelope" and not x["within_training_range"]
    assert any("clamped" in w for w in x["warnings"])


def test_clamping_bounds_output(c):
    a = p(c, {**DAY, "plant_id": 2, "irradiation": 1.45, "timestamp": "2020-06-07T12:00:00"}).json()["expected_ac_power"]
    b = p(c, {**DAY, "plant_id": 2, "irradiation": 1.10, "timestamp": "2020-06-07T12:00:00"}).json()["expected_ac_power"]
    assert abs(a - b) < 0.02 * b


# ---- consistency and season warnings ----
def test_cold_module_in_sun_warns(c):
    assert any("far from" in w for w in p(c, {**DAY, "plant_id": 1, "module_temperature": 20, "ambient_temperature": 35}).json()["warnings"])


def test_normal_row_has_no_warnings(c):
    x = p(c, {**DAY, "plant_id": 2}).json()
    assert x["warnings"] == [] and x["reliability"] == "ok"


def test_outside_season_warns(c):
    assert any("season" in w for w in p(c, {**DAY, "plant_id": 1, "timestamp": "2021-12-07T10:00:00"}).json()["warnings"])


# ---- rejected inputs (422) with helpful messages ----
@pytest.mark.parametrize("override,hint", [
    ({"irradiation": 720}, "W/m2"), ({"irradiation": -1}, "negative"),
    ({"module_temperature": 321.6}, "Kelvin"), ({"ambient_temperature": 304.4}, "Kelvin"),
    ({"ambient_temperature": 88}, "Fahrenheit"), ({"module_temperature": 120}, "Fahrenheit"),
    ({"ambient_temperature": -20}, "physical"),
    ({"timestamp": "2020-06-07"}, "time of day"), ({"timestamp": 1591524000}, "ISO"),
    ({"timestamp": "2020-06-07T02:00:00"}, "no sun"), ({"timestamp": "2020-06-07T21:30:00"}, "no sun"),
])
def test_rejected_with_hint(c, override, hint):
    r = p(c, {**DAY, "plant_id": 1, **override})
    assert r.status_code == 422 and hint in str(r.json()["detail"])


@pytest.mark.parametrize("override", [{"irradiation": True}, {"irradiation": "0.5"}, {"module_temperature": "48"},
                                      {"ambient_temperature": False}, {"plant_id": "1"}, {"irradiation": None}])
def test_wrong_types_rejected(c, override):
    assert p(c, {**DAY, "plant_id": 1, **override}).status_code == 422


def test_integers_accepted(c):
    assert p(c, {**DAY, "plant_id": 1, "irradiation": 1, "module_temperature": 55, "ambient_temperature": 30,
                 "timestamp": "2020-06-07T12:00:00"}).status_code == 200


# ---- receipts cover the rules ----
def test_receipt_roundtrip_with_missing_temp(c):
    body = {**DAY, "plant_id": 2, "module_temperature": None}
    o = p(c, body).json()
    assert o["rules_version"]
    assert c.post("/verify", json={**body, "expected_ac_power": o["expected_ac_power"], "receipt": o["receipt"]}).status_code == 200


def test_receipt_roundtrip_dark(c):
    o = p(c, {**NIGHT, "plant_id": 2}).json()
    assert c.post("/verify", json={**NIGHT, "plant_id": 2, "expected_ac_power": 0, "receipt": o["receipt"]}).status_code == 200


def test_batch_mixed_regimes(c):
    rs = [{**DAY, "plant_id": 1}, {**NIGHT, "plant_id": 2}, {**DAY, "plant_id": 2, "ambient_temperature": None}]
    out = c.post("/predict/batch", json={"readings": rs}).json()["results"]
    assert [x["regime"] for x in out] == ["daylight", "dark", "daylight"] and out[1]["expected_ac_power"] == 0
    assert [x["expected_ac_power"] for x in out] == [p(c, r).json()["expected_ac_power"] for r in rs]


def test_model_info_lists_rules(c):
    j = c.get("/model-info").json(); assert j["rules_version"] and len(j["rules"]) >= 5
