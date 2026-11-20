# Solar Expected Output API
FastAPI service for KMIT Project School PS5. It estimates expected plant AC output.

## Local
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements-dev.txt
    PYTHONPATH=. pytest -q
    uvicorn app.main:app --reload
Open http://localhost:8000/docs

## Retrain
Put the four dataset CSVs in data/, run `python ml/train.py`, run tests, and commit models/, tests/golden.json and FINGERPRINT.txt. Regenerate the Postman known row if golden values change.

## Endpoints
GET /health, GET /model-info, POST /predict, POST /predict/batch, POST /verify

## Settings
API_KEYS comma-separated keys, empty means open. CORS_ORIGINS comma-separated frontend URLs. MAX_BATCH default 500.

## Input edge-case rules (rules_version edge-rules-1)
- **Dark:** irradiation <= 0.02 (including small negative offsets down to -0.05) returns exactly 0, `regime: "dark"`; the model is not called, so temperatures do not matter.
- **Missing temperature:** send `null` (exactly `0` is treated the same, as a logger default). In daylight it is inferred: module = ambient + 31.9*I - 1.59 (fit on Plant 1). One inferred: `reliability: "reduced"`. Both missing: ambient = midpoint of the plant's training range, `reliability: "low"`. The field must still be present.
- **Outside training range:** each input is clamped to the plant's training range before predicting; `reliability: "outside_envelope"`, `within_training_range: false`.
- **Warnings only:** module temperature more than 15 C from the value implied by ambient and irradiation; module much hotter than ambient while dark; date outside 15 May - 17 June.
- **422 with a hint:** irradiation above 1.5 (W/m2 hint) or below -0.05; temperatures outside physical limits (Kelvin / Fahrenheit hints); sun above 0.05 between 20:00 and 05:00; timestamp without a time of day; booleans or strings instead of numbers.
- Every response carries `regime`, `reliability`, `inputs_used` and `rules_version`; receipts include `rules_version`, so receipts issued before edge-rules-1 no longer verify.
