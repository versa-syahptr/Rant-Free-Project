"""
db.py — MongoDB queue operations and Feast enrichment for HITL Backend.

Queue state (pending_reviews) lives in the same MongoDB instance as Feast,
in a separate collection defined by MONGODB_HITL_COLLECTION (default: "hitl_queue").
"""

import logging
import os
import pandas as pd
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection

from feature_store.store import get_store
from models import FeastEnrichment, QueueItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MongoDB queue collection
# ---------------------------------------------------------------------------

def get_queue_collection() -> Collection:
    client = MongoClient(os.environ["MONGODB_CONNECTION_STRING"])
    db = client[os.environ.get("MONGODB_DATABASE", "feast")]
    return db[os.environ.get("MONGODB_HITL_COLLECTION", "hitl_queue")]


def init_db() -> None:
    """Ensure index on request_id — idempotent, safe to call on every startup."""
    col = get_queue_collection()
    col.create_index([("request_id", ASCENDING)], unique=True)
    col.create_index([("status", ASCENDING), ("flagged_at", ASCENDING)])
    logger.info("hitl_queue collection indexes ensured")


# ---------------------------------------------------------------------------
# Feast operations
# ---------------------------------------------------------------------------

def fetch_feast_enrichment(request_id: str) -> Optional[FeastEnrichment]:
    """Fetch text + scores for a request_id from the Feast offline store."""
    try:
        store = get_store()
        entity_df = pd.DataFrame({
            "request_id":      [request_id],
            "event_timestamp": [datetime.now(timezone.utc)],
        })
        df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "comment_features:text",
                "comment_features:language",
                "comment_features:score_toxic",
                "comment_features:confidence",
                "comment_features:model_version",
            ],
        ).to_df()

        if df.empty:
            logger.warning("feast enrichment: no data found for request_id=%s", request_id)
            return None

        row = df.iloc[0]
        return FeastEnrichment(
            text=row.get("comment_features__text"),
            language=row.get("comment_features__language"),
            score_toxic=row.get("comment_features__score_toxic"),
            confidence=row.get("comment_features__confidence"),
            model_version=row.get("comment_features__model_version"),
        )
    except Exception:
        logger.exception("feast enrichment failed for request_id=%s", request_id)
        return None


def write_feast_verified_label(request_id: str, toxic: int) -> None:
    """
    Append a new row to Feast with the verified label filled in.
    Carries forward text, language, score_toxic, confidence, model_version
    from the original prediction row. Newer event_timestamp ensures this row
    wins over the prediction row when Training Service reads latest per entity.
    """
    try:
        store = get_store()

        entity_df = pd.DataFrame({
            "request_id":      [request_id],
            "event_timestamp": [datetime.now(timezone.utc)],
        })
        existing = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "comment_features:text",
                "comment_features:language",
                "comment_features:score_toxic",
                "comment_features:confidence",
                "comment_features:model_version",
            ],
        ).to_df()

        if existing.empty:
            logger.error(
                "feast write: request_id=%s not found, cannot write verified label",
                request_id,
            )
            return

        row = existing.iloc[0]
        df = pd.DataFrame([{
            "request_id":      request_id,
            "event_timestamp": datetime.now(timezone.utc),  # must be newer than prediction row
            "text":            row.get("comment_features__text"),
            "language":        row.get("comment_features__language"),
            "score_toxic":     row.get("comment_features__score_toxic"),
            "confidence":      row.get("comment_features__confidence"),
            "model_version":   row.get("comment_features__model_version"),
            "toxic":           toxic,
        }])
        store.write_to_offline_store(feature_view_name="comment_features", df=df)
        logger.info(
            "feast write: verified label written for request_id=%s toxic=%d",
            request_id, toxic,
        )
    except Exception:
        logger.exception("feast write failed for request_id=%s", request_id)


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

def enqueue(request_id: str, confidence: float) -> bool:
    """
    Insert a new pending review into the queue.
    Returns True if inserted, False if request_id already exists (idempotent).
    """
    col = get_queue_collection()
    doc = {
        "request_id":  request_id,
        "confidence":  confidence,
        "flagged_at":  datetime.now(timezone.utc),
        "status":      "pending",
        "reviewed_by": None,
        "reviewed_at": None,
    }
    try:
        col.insert_one(doc)
        logger.info("enqueued request_id=%s confidence=%.4f", request_id, confidence)
        return True
    except Exception:
        # duplicate key error — request_id already exists
        logger.warning("duplicate enqueue ignored for request_id=%s", request_id)
        return False


def get_pending_queue() -> list[QueueItem]:
    """
    Return all pending (unreviewed) items, enriched with Feast data.
    Ordered by flagged_at ascending (oldest first for reviewers).
    """
    col = get_queue_collection()
    docs = list(col.find({"status": "pending"}).sort("flagged_at", ASCENDING))

    items = []
    for doc in docs:
        feast_data = fetch_feast_enrichment(doc["request_id"])
        items.append(QueueItem(
            request_id=doc["request_id"],
            confidence=doc["confidence"],
            flagged_at=doc["flagged_at"],
            status=doc["status"],
            reviewed_by=doc.get("reviewed_by"),
            reviewed_at=doc.get("reviewed_at"),
            feast=feast_data,
        ))

    logger.info("get_pending_queue returned %d items", len(items))
    return items


def submit_review(request_id: str, reviewed_by: str, toxic: int) -> bool:
    """
    Mark a pending review as reviewed in MongoDB, then write verified
    label to Feast. Returns True if updated, False if request_id not
    found or already reviewed.
    """
    col = get_queue_collection()
    result = col.update_one(
        {"request_id": request_id, "status": "pending"},
        {"$set": {
            "status":      "reviewed",
            "reviewed_by": reviewed_by,
            "reviewed_at": datetime.now(timezone.utc),
        }},
    )

    if result.matched_count == 0:
        logger.warning(
            "submit_review: request_id=%s not found or already reviewed", request_id
        )
        return False

    logger.info(
        "submit_review: request_id=%s reviewed_by=%s toxic=%d",
        request_id, reviewed_by, toxic,
    )
    write_feast_verified_label(request_id, toxic)
    return True