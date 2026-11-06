"""Solar Expected Output API. Output is an estimate for running inverters,
not a forecast and never a fault diagnosis."""
import hmac, json, logging, sys, time, uuid
from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .schemas import Batch, Prediction, Reading, VerifyRequest
from .service import ModelService

log = logging.getLogger("solar"); log.setLevel(logging.INFO)
if not log.handlers:
    log.addHandler(logging.StreamHandler(sys.stdout))

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.svc = ModelService()
    yield

app = FastAPI(title="Solar Expected Output API", version="1.0.0", lifespan=lifespan)
if settings.cors_origins:
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"], allow_headers=["*"])

@app.middleware("http")
async def access_log(request: Request, call_next):
    rid, start = request.headers.get("x-request-id", uuid.uuid4().hex[:12])[:64], time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled path=%s rid=%s", request.url.path, rid)
        response = JSONResponse({"detail": "internal error", "request_id": rid}, status_code=500)
    response.headers["x-request-id"] = rid
    log.info(json.dumps({"rid": rid, "method": request.method, "path": request.url.path, "status": response.status_code,
                         "ms": round((time.perf_counter()-start)*1000, 1)}))
    return response

def auth(x_api_key: str | None = Header(default=None)):
    if settings.api_keys and not any(hmac.compare_digest(x_api_key or "", k) for k in settings.api_keys):
        raise HTTPException(401, "missing or invalid API key")

def svc(request: Request) -> ModelService:
    return request.app.state.svc

@app.get("/", include_in_schema=False)
def root(): return {"service": "Solar Expected Output API", "docs": "/docs"}

@app.get("/health")
def health(s=Depends(svc)): return {"status": "ok", "fingerprint": s.fingerprint}

@app.get("/model-info")
def model_info(s=Depends(svc)):
    m = s.meta
    return {"fingerprint": s.fingerprint, "unit": "source AC units, plant total", "plants": {k: m[k] for k in ("1", "2")},
            "manifest": m["manifest"], "note": "Estimate if inverters are running; not a forecast or fault diagnosis."}

@app.post("/predict", response_model=Prediction, dependencies=[Depends(auth)])
def predict(r: Reading, s=Depends(svc)):
    return s.present(r, s.estimate_many([r])[0])

@app.post("/predict/batch", dependencies=[Depends(auth)])
def predict_batch(b: Batch, s=Depends(svc)):
    if len(b.readings) > settings.max_batch:
        raise HTTPException(413, f"maximum {settings.max_batch} readings")
    return {"results": [s.present(r, v) for r, v in zip(b.readings, s.estimate_many(b.readings))]}

@app.post("/verify", dependencies=[Depends(auth)])
def verify(v: VerifyRequest, s=Depends(svc)):
    r = Reading(**v.model_dump(include=set(Reading.model_fields)))
    ok, recomputed = s.verify(r, v.expected_ac_power, v.receipt)
    if not ok:
        raise HTTPException(409, {"verified": False, "recomputed_expected_ac_power": recomputed})
    return {"verified": True, "fingerprint": s.fingerprint}
