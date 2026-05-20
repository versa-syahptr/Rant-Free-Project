import asyncio
import collections
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

VOLUME_WINDOW = 60
HEALTH_POLL_INTERVAL = int(os.getenv("HEALTH_POLL_INTERVAL", "30"))

SERVICES = {
    "model-service": os.getenv("MODEL_SERVICE_URL", "http://localhost:8000"),
    "hitl-backend":  os.getenv("HITL_BACKEND_URL",  "http://localhost:8002"),
}


# --- prediction event store ---

class _Store:
    def __init__(self):
        self.total = 0
        self.score_toxic_sum: float = 0.0
        self.confidence_sum = 0.0
        self.timestamps: collections.deque[float] = collections.deque()
        self.alerts: list[dict] = []

    def record(self, event: dict):
        now = time.time()
        self.total += 1
        self.confidence_sum += event["confidence"]
        self.score_toxic_sum += event["score_toxic"]
        self.timestamps.append(now)
        cutoff = now - VOLUME_WINDOW
        while self.timestamps and self.timestamps[0] < cutoff:
            self.timestamps.popleft()

    def add_alert(self, event: dict):
        self.alerts.append(event)
        if len(self.alerts) > 200:
            self.alerts.pop(0)


_store = _Store()


# --- service health store ---

class _CheckResult:
    __slots__ = ("ts", "status", "latency_ms")

    def __init__(self, ts: float, status: str, latency_ms: float):
        self.ts = ts
        self.status = status
        self.latency_ms = latency_ms


class _ServiceHealth:
    HISTORY_SIZE = 100

    def __init__(self, name: str, url: str):
        self.name = name
        self.url = url
        self.status: str = "unknown"
        self.last_checked: Optional[float] = None
        self.latency_ms: Optional[float] = None
        self.consecutive_failures: int = 0
        self._history: collections.deque[_CheckResult] = collections.deque(
            maxlen=self.HISTORY_SIZE
        )

    def record(self, ok: bool, latency_ms: float):
        now = time.time()
        self.status = "up" if ok else "down"
        self.last_checked = now
        self.latency_ms = latency_ms if ok else None
        self.consecutive_failures = 0 if ok else self.consecutive_failures + 1
        self._history.append(_CheckResult(now, self.status, latency_ms))

    @property
    def uptime_pct(self) -> Optional[float]:
        if not self._history:
            return None
        up = sum(1 for h in self._history if h.status == "up")
        return round(up / len(self._history) * 100, 1)

    def recent_checks(self, n: int = 20) -> list[dict]:
        checks = list(self._history)[-n:]
        return [
            {"ts": c.ts, "status": c.status, "latency_ms": round(c.latency_ms, 2)}
            for c in checks
        ]


_health: dict[str, _ServiceHealth] = {
    name: _ServiceHealth(name, url) for name, url in SERVICES.items()
}
_http_client = httpx.AsyncClient(timeout=5.0)


async def _poll_health():
    while True:
        for svc in _health.values():
            t0 = time.perf_counter()
            try:
                resp = await _http_client.get(f"{svc.url}/health")
                latency_ms = (time.perf_counter() - t0) * 1000
                ok = resp.status_code == 200
            except Exception as exc:
                latency_ms = (time.perf_counter() - t0) * 1000
                ok = False
                logger.warning(f"Health check failed for {svc.name}: {exc}")
            svc.record(ok, latency_ms)
            logger.info(f"[health] {svc.name} → {svc.status} ({latency_ms:.0f} ms)")
        await asyncio.sleep(HEALTH_POLL_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_poll_health())
    yield
    task.cancel()
    await _http_client.aclose()


# --- app ---

app = FastAPI(title="Rant-Free Monitoring Service", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- schemas ---

class PredictionEvent(BaseModel):
    request_id: str
    text: str
    score_toxic: float
    confidence: float
    model_version: str
    low_confidence: bool = False


class MetricsResponse(BaseModel):
    total_predictions: int
    volume_last_minute: int
    avg_confidence: float
    avg_score_toxic: float
    services: dict[str, str]  # name → "up" | "down" | "unknown"


class AlertItem(BaseModel):
    request_id: str
    text: str
    confidence: float
    score_toxic: float


class ServiceStatusResponse(BaseModel):
    name: str
    url: str
    status: str
    last_checked: Optional[float]
    latency_ms: Optional[float]
    consecutive_failures: int
    uptime_pct: Optional[float]


class ServiceDetailResponse(ServiceStatusResponse):
    recent_checks: list[dict]


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
    n = _store.total or 1
    return MetricsResponse(
        total_predictions=_store.total,
        volume_last_minute=len(_store.timestamps),
        avg_confidence=_store.confidence_sum / n,
        avg_score_toxic=_store.score_toxic_sum / n,
        services={name: svc.status for name, svc in _health.items()},
    )


@app.get("/alerts", response_model=list[AlertItem])
async def alerts(limit: int = 50):
    return _store.alerts[-limit:]


@app.get("/services", response_model=list[ServiceStatusResponse])
async def list_services():
    return [
        ServiceStatusResponse(
            name=svc.name,
            url=svc.url,
            status=svc.status,
            last_checked=svc.last_checked,
            latency_ms=svc.latency_ms,
            consecutive_failures=svc.consecutive_failures,
            uptime_pct=svc.uptime_pct,
        )
        for svc in _health.values()
    ]


@app.get("/services/{name}", response_model=ServiceDetailResponse)
async def service_detail(name: str):
    svc = _health.get(name)
    if not svc:
        raise HTTPException(status_code=404, detail=f"Service '{name}' not registered")
    return ServiceDetailResponse(
        name=svc.name,
        url=svc.url,
        status=svc.status,
        last_checked=svc.last_checked,
        latency_ms=svc.latency_ms,
        consecutive_failures=svc.consecutive_failures,
        uptime_pct=svc.uptime_pct,
        recent_checks=svc.recent_checks(),
    )
