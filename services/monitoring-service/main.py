import collections
import logging
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Rolling window for volume tracking (seconds)
VOLUME_WINDOW = 60

app = FastAPI(title="Rant-Free Monitoring Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- in-memory store ---

class _Store:
    def __init__(self):
        self.total = 0
        self.score_sums: dict[str, float] = {l: 0.0 for l in LABELS}
        self.confidence_sum = 0.0
        # timestamps of recent requests for volume-per-minute
        self.timestamps: collections.deque[float] = collections.deque()
        # low-confidence events queued for alert
        self.alerts: list[dict] = []

    def record(self, event: dict):
        now = time.time()
        self.total += 1
        self.confidence_sum += event["confidence"]
        for label in LABELS:
            self.score_sums[label] += event["scores"].get(label, 0.0)
        self.timestamps.append(now)
        # prune old timestamps outside the window
        cutoff = now - VOLUME_WINDOW
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def add_alert(self, event: dict):
        self.alerts.append(event)
        if len(self.alerts) > 200:
            self.alerts.pop(0)


_store = _Store()


# --- schemas ---

class PredictionEvent(BaseModel):
    request_id: str
    text: str
    scores: dict[str, float]
    confidence: float
    model_version: str
    low_confidence: bool = False


class MetricsResponse(BaseModel):
    total_predictions: int
    volume_last_minute: int
    avg_confidence: float
    avg_scores: dict[str, float]


class AlertItem(BaseModel):
    request_id: str
    text: str
    confidence: float
    scores: dict[str, float]


# --- endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/events", status_code=204)
async def receive_event(event: PredictionEvent):
    _store.record(event.model_dump())
    if event.low_confidence:
        _store.add_alert(event.model_dump())
    logger.info(f"Event recorded [{event.request_id}] confidence={event.confidence:.4f}")


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    n = _store.total or 1  # avoid div-by-zero
    return MetricsResponse(
        total_predictions=_store.total,
        volume_last_minute=len(_store.timestamps),
        avg_confidence=_store.confidence_sum / n,
        avg_scores={l: _store.score_sums[l] / n for l in LABELS},
    )


@app.get("/alerts", response_model=list[AlertItem])
async def alerts(limit: int = 50):
    return _store.alerts[-limit:]
