"""Heart disease risk API.

Endpoints:
    GET  /          service info
    GET  /health    readiness/liveness (503 until the model is loaded)
    POST /predict   JSON patient record -> prediction + confidence
    GET  /metrics   Prometheus metrics

Run locally:
    uvicorn api.main:app --reload
"""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field

from src.predict import load_metadata, load_model, model_path, predict_one

# ---------------------------------------------------------------- logging
logger = logging.getLogger("heart_api")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_event(event: str, **fields):
    """One JSON object per line: easy to grep and to ship to a log stack."""
    logger.info(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event,
                            **fields}))


# ---------------------------------------------------------------- metrics
REQUESTS = Counter("api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions served", ["label"])
KNOWN_ENDPOINTS = {"/", "/health", "/predict", "/docs", "/openapi.json"}
DISEASE_PROB = Histogram("model_disease_probability", "Predicted P(disease)",
                         buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))


# ---------------------------------------------------------------- schemas
class PatientRecord(BaseModel):
    """Raw UCI coding. ca and thal may be omitted; the pipeline imputes them."""
    age: float = Field(..., ge=18, le=110, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type")
    trestbps: float = Field(..., ge=60, le=250, description="Resting BP (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels (0-3)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed, 7 reversible")

    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
        "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 2.4, "slope": 2,
        "ca": 2, "thal": 7}})


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # allow model_name/model_version

    prediction: int
    label: str
    probability_disease: float
    confidence: float
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    request_id: str


# ---------------------------------------------------------------- app
state: dict = {"model": None, "metadata": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = model_path()
    try:
        state["model"] = load_model(path)
        state["metadata"] = load_metadata(path)
        log_event("model_loaded", path=str(path),
                  model=state["metadata"].get("model_name"),
                  version=state["metadata"].get("model_version"))
    except FileNotFoundError:
        log_event("model_missing", path=str(path))
    yield
    state["model"] = None


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    path = request.url.path
    if path.startswith("/metrics"):
        return await call_next(request)

    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    request.state.request_id = request_id
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        endpoint = path if path in KNOWN_ENDPOINTS else "other"  # bound label cardinality
        REQUESTS.labels(request.method, endpoint, str(status)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        log_event("request", request_id=request_id, method=request.method, path=path,
                  status=status, latency_ms=round(elapsed * 1000, 2),
                  client=request.client.host if request.client else None)


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "docs": "/docs",
            "model": state["metadata"].get("model_name"),
            "model_version": state["metadata"].get("model_version")}


@app.get("/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model_version": state["metadata"].get("model_version")}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: PatientRecord, request: Request):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    result = predict_one(state["model"], record.model_dump())
    PREDICTIONS.labels(result["label"]).inc()
    DISEASE_PROB.observe(result["probability_disease"])
    # Only the outcome is logged, never the patient's clinical values.
    log_event("prediction", request_id=request.state.request_id,
              prediction=result["prediction"],
              probability_disease=result["probability_disease"],
              imputed=[f for f in ("ca", "thal") if getattr(record, f) is None])
    return PredictionResponse(**result, request_id=request.state.request_id,
                              model_name=state["metadata"].get("model_name"),
                              model_version=state["metadata"].get("model_version"))
