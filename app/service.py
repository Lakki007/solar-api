from __future__ import annotations
import hashlib, hmac, json, math
import joblib, pandas as pd
from .config import settings
from .schemas import Reading

UNIT = "source AC units, plant total"
RAW = ("irradiation", "module_temperature", "ambient_temperature")

class ModelService:
    def __init__(self):
        self.fingerprint = hashlib.sha256(settings.model_path.read_bytes()).hexdigest()
        self.bundle = joblib.load(settings.model_path)
        self.meta = json.loads(settings.metrics_path.read_text())
        if self.meta["manifest"]["model_sha256"] != self.fingerprint:
            raise RuntimeError("model file does not match its manifest")

    def _x(self, rs):
        rows = []
        for r in rs:
            h = r.timestamp.hour + r.timestamp.minute/60
            rows.append([r.irradiation, r.module_temperature, r.ambient_temperature,
                         math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24)])
        return pd.DataFrame(rows, columns=self.bundle["features"])

    def estimate_many(self, rs: list[Reading]) -> list[float]:
        out = [0.0]*len(rs)
        for pid in {r.plant_id for r in rs}:
            idx = [i for i, r in enumerate(rs) if r.plant_id == pid]
            ys = self.bundle["models"][pid].predict(self._x([rs[i] for i in idx]))
            for i, y in zip(idx, ys):
                out[i] = round(max(0.0, float(y)), 2)
        return out

    def receipt(self, r: Reading, value: float) -> str:
        body = {"plant_id": r.plant_id, **{k: getattr(r, k) for k in RAW}, "timestamp": r.timestamp.isoformat(),
                "expected_ac_power": value, "fingerprint": self.fingerprint}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def warnings(self, r: Reading) -> list[str]:
        rng = self.meta[str(r.plant_id)]["ranges"]
        w = [f"{k} is outside the training range" for k in RAW if not rng[k][0] <= getattr(r, k) <= rng[k][1]]
        if r.irradiation > .05 and not 5 <= r.timestamp.hour <= 19:
            w.append("sunlight reported at an unusual hour; check the timestamp or sensor")
        return w

    def present(self, r: Reading, value: float) -> dict:
        w = self.warnings(r)
        return {"expected_ac_power": value, "unit": UNIT, "model": self.meta[str(r.plant_id)]["chosen"],
                "fingerprint": self.fingerprint, "receipt": self.receipt(r, value),
                "within_training_range": not any("training range" in x for x in w), "warnings": w}

    def verify(self, r: Reading, claimed: float, receipt: str):
        v = self.estimate_many([r])[0]
        return (v == round(claimed, 2) and hmac.compare_digest(self.receipt(r, v), receipt)), v
