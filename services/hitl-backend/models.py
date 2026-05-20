from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Pydantic — request / response schemas
# ---------------------------------------------------------------------------


class EnqueueRequest(BaseModel):
    """Payload sent by Model Service when confidence < threshold."""
    request_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class EnqueueResponse(BaseModel):
    request_id: str
    status: str = "queued"


class FeastEnrichment(BaseModel):
    """
    Data fetched from Feast to enrich queue items for the reviewer UI.
    All fields are Optional — gracefully handles the case where Feast
    has not yet written data for this request_id.
    """
    text: Optional[str] = None
    language: Optional[str] = None
    score_toxic: Optional[float] = None
    confidence: Optional[float] = None
    model_version: Optional[str] = None


class QueueItem(BaseModel):
    """Single item returned by GET /queue — queue state + Feast enrichment."""
    request_id: str
    confidence: float
    flagged_at: datetime
    status: str
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    feast: Optional[FeastEnrichment] = None


class QueueResponse(BaseModel):
    items: list[QueueItem]
    total: int


class ReviewRequest(BaseModel):
    """Payload sent by HITL UI when a reviewer submits a verified label."""
    reviewed_by: str
    toxic: int = Field(..., ge=0, le=1)


class ReviewResponse(BaseModel):
    request_id: str
    status: str = "reviewed"