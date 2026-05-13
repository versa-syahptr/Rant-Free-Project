"""
HITL Backend — FastAPI service
Endpoints:
    POST /queue                     ← called by Model Service (fire-and-forget)
    GET  /queue                     ← called by HITL UI
    POST /review/{request_id}       ← called by HITL UI
    GET  /health                    ← liveness probe
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from db import enqueue, get_pending_queue, init_db, submit_review
from models import (
    EnqueueRequest,
    EnqueueResponse,
    QueueResponse,
    ReviewRequest,
    ReviewResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("hitl-backend")

# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    logger.info("HITL Backend ready")
    yield


app = FastAPI(
    title="HITL Backend",
    description="Human-in-the-loop review queue for Rant-Free content moderation",
    version="0.1.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["ops"])
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/queue", response_model=EnqueueResponse, status_code=202, tags=["queue"])
def post_queue(body: EnqueueRequest):
    """
    Called by Model Service (fire-and-forget) when confidence < threshold.
    Idempotent — duplicate request_ids are silently ignored.
    """
    enqueue(body.request_id, body.confidence)
    return EnqueueResponse(request_id=body.request_id)


@app.get("/queue", response_model=QueueResponse, tags=["queue"])
def get_queue():
    """
    Returns all pending (unreviewed) items for the HITL UI.
    Each item is enriched with text + scores from Feast.
    """
    items = get_pending_queue()
    return QueueResponse(items=items, total=len(items))


@app.post("/review/{request_id}", response_model=ReviewResponse, tags=["review"])
def post_review(request_id: str, body: ReviewRequest):
    """
    Called by HITL UI when reviewer submits a verified label.
    - Updates queue status to 'reviewed' in MongoDB
    - Writes verified label to Feast
    Returns 404 if request_id not found or already reviewed.
    """
    updated = submit_review(request_id, body.reviewed_by, body.toxic)
    if not updated:
        raise HTTPException(
            status_code=404,
            detail=f"request_id '{request_id}' not found or already reviewed",
        )
    return ReviewResponse(request_id=request_id)