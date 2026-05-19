import time
import uuid
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Rant-Free HITL Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- in-memory queue ---

class QueueItem(BaseModel):
    item_id: str
    request_id: str
    text: str
    confidence: float
    scores: dict[str, float]
    queued_at: float
    status: Literal["pending", "approved", "rejected"] = "pending"
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[float] = None


_queue: dict[str, QueueItem] = {}


# --- schemas ---

class EnqueueRequest(BaseModel):
    request_id: str
    text: str
    confidence: float
    scores: dict[str, float]

class ReviewRequest(BaseModel):
    decision: Literal["approved", "rejected"]
    reviewer: str = "human"


# --- endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/queue", status_code=201)
async def enqueue(req: EnqueueRequest):
    item_id = str(uuid.uuid4())
    _queue[item_id] = QueueItem(
        item_id=item_id,
        request_id=req.request_id,
        text=req.text,
        confidence=req.confidence,
        scores=req.scores,
        queued_at=time.time(),
    )
    return {"item_id": item_id}


@app.get("/queue", response_model=list[QueueItem])
async def list_queue(status: Literal["pending", "approved", "rejected"] = "pending"):
    return [item for item in _queue.values() if item.status == status]


@app.post("/queue/{item_id}/review", response_model=QueueItem)
async def review(item_id: str, req: ReviewRequest):
    item = _queue.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    if item.status != "pending":
        raise HTTPException(status_code=409, detail=f"Item already {item.status}")
    item.status = req.decision
    item.reviewed_by = req.reviewer
    item.reviewed_at = time.time()
    return item
