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
