from __future__ import annotations
import hashlib, hmac, json, math
import joblib, pandas as pd
from .config import settings
from .schemas import Reading

UNIT = "source AC units, plant total"
RAW = ("irradiation", "module_temperature", "ambient_temperature")
RULES_VERSION = "edge-rules-1"

DARK = 0.02                     # same dark threshold train.py uses for its labels
K_HEAT, C_HEAT = 31.9, -1.59    # module ~= ambient + 31.9*I - 1.59 (fit on Plant 1)
TEMP_MISMATCH = 15.0            # deg C between measured and implied module temperature
DARK_HOT = 10.0                 # module this much hotter than ambient while dark is suspicious
SEASON = ((5, 15), (6, 17))     # training window, month/day (15 May - 17 June 2020)

RULES = [
    "Physically impossible or unit-mistaken inputs are rejected with 422 and a hint.",
    f"irradiation <= {DARK} is dark: expected output is 0 and the model is not called.",
    "In daylight a missing temperature (null or exactly 0) is inferred: module = ambient + 31.9*I - 1.59.",
    "If both temperatures are missing, ambient = midpoint of the plant's training range; reliability low.",
    "Inputs outside the plant's training range are clamped to it before predicting; reliability outside_envelope.",
    "Inconsistent temperatures and dates outside the training season give warnings.",
]


def _missing(v):
    return v is None or v == 0.0


class ModelService:
    def __init__(self):
        self.fingerprint = hashlib.sha256(settings.model_path.read_bytes()).hexdigest()
        self.bundle = joblib.load(settings.model_path)
        self.meta = json.loads(settings.metrics_path.read_text())
        if self.meta["manifest"]["model_sha256"] != self.fingerprint:
            raise RuntimeError("model file does not match its manifest")

    # ---------- edge-case rules ----------
    def evaluate(self, r: Reading) -> dict:
        """Decide regime, the inputs actually fed to the model, reliability and warnings."""
        rng = self.meta[str(r.plant_id)]["ranges"]
        I, mod, amb = r.irradiation, r.module_temperature, r.ambient_temperature
        w: list[str] = []
        md = (r.timestamp.month, r.timestamp.day)
        if not SEASON[0] <= md <= SEASON[1]:
            w.append("date is outside the training season (15 May - 17 June); sun angle and day length may differ")

        if I <= DARK:
            if I < 0:
                w.append("small negative irradiation treated as 0 (sensor night offset)")
            if not _missing(mod) and not _missing(amb) and mod - amb > DARK_HOT:
                w.append("module is much hotter than ambient while dark; check the temperature sensors or timestamp")
            used = {"irradiation": max(I, 0.0), "module_temperature": mod, "ambient_temperature": amb}
            return {"regime": "dark", "used": used, "reliability": "ok", "in_range": True, "warnings": w}

        inferred = 0
        if _missing(mod) and _missing(amb):
            lo, hi = rng["ambient_temperature"]
            amb = round((lo + hi) / 2, 2)
            mod = amb + K_HEAT * I + C_HEAT
            inferred = 2
            w.append(f"both temperatures missing (null or 0); ambient assumed {amb} C (training midpoint) and module inferred")
        elif _missing(mod):
            mod = amb + K_HEAT * I + C_HEAT
            inferred = 1
            w.append("module_temperature missing (null or 0); inferred from ambient and irradiation")
        elif _missing(amb):
            amb = mod - K_HEAT * I - C_HEAT
            inferred = 1
            w.append("ambient_temperature missing (null or 0); inferred from module and irradiation")
        else:
            implied = amb + K_HEAT * I + C_HEAT
            if abs(mod - implied) > TEMP_MISMATCH:
                w.append(f"module_temperature {mod} C is far from the {implied:.1f} C implied by ambient and irradiation; "
                         "check the sensors")

        used, clamped = {}, []
        for k, v in (("irradiation", I), ("module_temperature", mod), ("ambient_temperature", amb)):
            lo, hi = rng[k]
            c = min(max(v, lo), hi)
            if c != v:
                clamped.append(k)
                w.append(f"{k} {round(v, 2)} is outside the training range [{lo:.2f}, {hi:.2f}]; clamped to {c:.2f}")
            used[k] = c

        rel = "outside_envelope" if clamped else ("low" if inferred == 2 else "reduced" if inferred else "ok")
        return {"regime": "daylight", "used": used, "reliability": rel, "in_range": not clamped, "warnings": w}

    # ---------- model ----------
    def _x(self, rs, used):
        rows = []
        for r, u in zip(rs, used):
            h = r.timestamp.hour + r.timestamp.minute/60
            rows.append([u["irradiation"], u["module_temperature"], u["ambient_temperature"],
                         math.sin(2*math.pi*h/24), math.cos(2*math.pi*h/24)])
        return pd.DataFrame(rows, columns=self.bundle["features"])

    def estimate_many(self, rs: list[Reading]) -> list[tuple[float, dict]]:
        evs = [self.evaluate(r) for r in rs]
        out = [0.0]*len(rs)
        for pid in {r.plant_id for r in rs}:
            idx = [i for i, r in enumerate(rs) if r.plant_id == pid and evs[i]["regime"] == "daylight"]
            if not idx:
                continue
            ys = self.bundle["models"][pid].predict(self._x([rs[i] for i in idx], [evs[i]["used"] for i in idx]))
            for i, y in zip(idx, ys):
                out[i] = round(max(0.0, float(y)), 2)
        return list(zip(out, evs))

    def receipt(self, r: Reading, value: float) -> str:
        body = {"plant_id": r.plant_id, **{k: getattr(r, k) for k in RAW}, "timestamp": r.timestamp.isoformat(),
                "expected_ac_power": value, "fingerprint": self.fingerprint, "rules_version": RULES_VERSION}
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def present(self, r: Reading, value: float, ev: dict) -> dict:
        return {"expected_ac_power": value, "unit": UNIT, "model": self.meta[str(r.plant_id)]["chosen"],
                "regime": ev["regime"], "reliability": ev["reliability"],
                "inputs_used": {k: (None if v is None else round(v, 4)) for k, v in ev["used"].items()},
                "within_training_range": ev["in_range"], "warnings": ev["warnings"], "rules_version": RULES_VERSION,
                "fingerprint": self.fingerprint, "receipt": self.receipt(r, value)}

    def verify(self, r: Reading, claimed: float, receipt: str):
        v = self.estimate_many([r])[0][0]
        return (v == round(claimed, 2) and hmac.compare_digest(self.receipt(r, v), receipt)), v
