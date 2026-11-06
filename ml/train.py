"""Reproducible training for the Solar Expected Output API.
Run from the repo root with the four dataset CSVs in data/:  python ml/train.py

Target: expected plant AC if inverters are running
        = median output of running inverters x inverter count.
Split:  chronological 60/20/20 by day (train / calibration / test).
Selection uses calibration MAE only. Test data never influences a choice."""
from __future__ import annotations
import hashlib, json, platform
from datetime import datetime, timezone
from pathlib import Path
import joblib, numpy as np, pandas as pd, sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["irradiation", "module_temperature", "ambient_temperature", "hour_sin", "hour_cos"]
RAW = ["irradiation", "module_temperature", "ambient_temperature"]

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def frame(plant: int):
    gp, wp = ROOT/f"data/Plant_{plant}_Generation_Data.csv", ROOT/f"data/Plant_{plant}_Weather_Sensor_Data.csv"
    g, w = pd.read_csv(gp), pd.read_csv(wp)
    g["t"] = pd.to_datetime(g.DATE_TIME, dayfirst=plant == 1, format="mixed")
    w["t"] = pd.to_datetime(w.DATE_TIME, format="mixed")
    n = g.SOURCE_KEY.nunique()
    running = g[g.AC_POWER > 0].groupby("t").AC_POWER.median() * n
    w = w.drop_duplicates("t").set_index("t").sort_index()
    d = pd.DataFrame({"irradiation": w.IRRADIATION, "module_temperature": w.MODULE_TEMPERATURE,
                      "ambient_temperature": w.AMBIENT_TEMPERATURE}).join(running.rename("y"))
    d.loc[d.irradiation <= 0.02, "y"] = d.y.where(d.irradiation > 0.02).fillna(0)  # dark: zero is genuine
    d = d[d.y.notna() & d[RAW].notna().all(axis=1)]                                # never turn missing into zero
    h = d.index.hour + d.index.minute/60
    d["hour_sin"], d["hour_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    return d, n, [gp, wp]

def metrics(y, p):
    y = np.asarray(y)
    return {"mae": round(float(mean_absolute_error(y, p)), 2), "rmse": round(float(mean_squared_error(y, p)**.5), 2),
            "r2": round(float(r2_score(y, p)), 4), "daylight_mae": round(float(mean_absolute_error(y[y > 0], p[y > 0])), 2)}

def candidates():
    yield "linear", {}, LinearRegression()
    for leaf in (3, 10):
        yield "random_forest", {"min_samples_leaf": leaf}, RandomForestRegressor(150, min_samples_leaf=leaf, random_state=0, n_jobs=-1)
    for c in (500, 2000, 8000):
        yield "svr", {"C": c}, make_pipeline(StandardScaler(), SVR(C=c, epsilon=20))

def main():
    rep, bundle, data = {}, {}, {}
    for plant in (1, 2):
        d, n, paths = frame(plant)
        days = sorted(set(d.index.date)); a, b = days[int(len(days)*.6)], days[int(len(days)*.8)]
        dt = pd.Series(d.index.date, index=d.index)
        tr, ca, te = d[dt < a], d[(dt >= a) & (dt < b)], d[dt >= b]
        pred = lambda m, x: m.predict(x[FEATURES]).clip(0)
        best_by_family, runs = {}, []
        for fam, params, m in candidates():
            m.fit(tr[FEATURES], tr.y); c = float(mean_absolute_error(ca.y, pred(m, ca)))
            runs.append({"family": fam, "params": params, "calibration_mae": round(c, 2)})
            if fam not in best_by_family or c < best_by_family[fam][0]: best_by_family[fam] = (c, params, m)
        chosen = min(best_by_family, key=lambda k: best_by_family[k][0])
        models = {k: {"params": v[1], "calibration_mae": round(v[0], 2), "test": metrics(te.y, pred(v[2], te))}
                  for k, v in best_by_family.items()}
        base = te.assign(h=te.index.hour).h.map(tr.assign(h=tr.index.hour).groupby("h").y.mean()).fillna(0)
        m = best_by_family[chosen][2]; bundle[plant] = m
        rep[str(plant)] = {"chosen": chosen, "inverters": n, "rows": {"train": len(tr), "calibration": len(ca), "test": len(te)},
                           "calibration_start": str(a), "test_start": str(b), "search": runs, "models": models,
                           "baseline_hour_of_day_test": metrics(te.y, base.values),
                           "ranges": {c: [float(d[c].min()), float(d[c].max())] for c in RAW}}
        for p in paths: data[p.name] = sha(p)
        g = te[te.irradiation > .2].iloc[::40].head(8)
        rep[str(plant)]["golden"] = [{"plant_id": plant, "irradiation": float(r.irradiation), "module_temperature": float(r.module_temperature),
            "ambient_temperature": float(r.ambient_temperature), "timestamp": t.isoformat(), "expected_ac_power": round(float(v), 2)}
            for (t, r), v in zip(g.iterrows(), pred(m, g))]
    (ROOT/"models").mkdir(exist_ok=True); art = ROOT/"models/model.joblib"
    joblib.dump({"models": bundle, "features": FEATURES}, art, compress=3)
    fp = sha(art)
    golden = [x for p in "12" for x in rep[p].pop("golden")]
    rep["manifest"] = {"model_sha256": fp, "data_sha256": data, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}
    (ROOT/"models/metrics.json").write_text(json.dumps(rep, indent=2))
    (ROOT/"tests/golden.json").write_text(json.dumps(golden, indent=1))
    (ROOT/"FINGERPRINT.txt").write_text(fp + "\n")
    print(json.dumps({k: {"chosen": rep[k]["chosen"], "models": rep[k]["models"], "base": rep[k]["baseline_hour_of_day_test"]} for k in "12"}, indent=1))

if __name__ == "__main__":
    main()
