# Rant-Free Project - Model Service
# main.py - FastAPI app for serving the model
# Author: Versa and Abdi

import asyncio
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Literal
from contextlib import asynccontextmanager

from lang import detect_lang_with_fasttext
from model import RantFreeModel
from feast_writter import write_prediction_v2
from utils import get_info_from_env

logger = logging.getLogger("uvicorn.error")



# jigsaw dataset labels
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.3"))


_gpu_executor   = ThreadPoolExecutor(max_workers=1)
_feast_executor = ThreadPoolExecutor(max_workers=2)
_http_client    = httpx.AsyncClient()

use_dummy = os.getenv("USE_DUMMY") == "true"
if use_dummy:
    embedding_model_name_or_path = "dummy"
    classifier_model_path = "dummy"
else:
    embedding_model_name_or_path, classifier_model_path = get_info_from_env()

_model          = RantFreeModel(
    embedding_model_name_or_path=embedding_model_name_or_path,
    classifier_model_path=classifier_model_path,
)


# Request and Response Models
class PredictionRequest(BaseModel):
    """
    as simple as {text: "some text to classify"}
    """
    text: str = Field(..., description="The text to predict the class for")

    @field_validator('text')
    def validate_text(cls, value):
        if not value.strip():
            raise ValueError("Text cannot be empty")
        return value
    
class TokenReason(BaseModel):
    token: str
    score: Annotated[float, Field(le=0.0, ge=1.0)]
    
class PredictionResponse(BaseModel):
    """
    {score_toxic: float, reason: [{token: str, score: float}]}
    """
    score_toxic: float
    confidence:  float
    request_id:  str
    reason:      list[TokenReason]

class HealthResponse(BaseModel):
    status: str
    detail: Optional[str] = None

# --- helpers ---
def _compute_confidence(score: float) -> float:
    """
    Compute the confidence of the model's predictions.
    A score of 0.99 or 0.01 → model is very sure → contributes 0.49 to confidence
    A score of 0.51 or 0.49 → model is unsure → contributes 0.01 to confidence
    """
    return abs(score - 0.5)

async def _enqueue_hitl(request_id: str, confidence: float):
    try:
        # await _http_client.post(
        #     f"{HITL_BACKEND_URL}/queue",
        #     json={"request_id": request_id, "confidence": confidence},
        #     timeout=5.0,
        # )
        await asyncio.sleep(0.1)  # Simulate network delay
        logger.info(f"HITL enqueue successful [{request_id}] (confidence={confidence:.4f})")
    except Exception as e:
        logger.error(f"HITL enqueue failed [{request_id}]: {e}")

# --- lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    _model.start_watcher()
    yield
    await _http_client.aclose()


# --- app ---
app = FastAPI(
    title="Rant-Free Model Service",
    version="0.1.0",
    lifespan=lifespan,
)


# Health Check Endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy")

# Classification Endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    loop       = asyncio.get_event_loop()
    request_id = str(uuid.uuid4())

    score_toxic, reason = await loop.run_in_executor(_gpu_executor, _model.predict, request.text)
    confidence = _compute_confidence(score_toxic)
    language = detect_lang_with_fasttext(request.text)

    loop.run_in_executor(
        _feast_executor,
        write_prediction_v2,
        request_id, request.text, score_toxic, confidence, language, _model.version,
    )

    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(f"Low confidence ({confidence:.4f}) for request {request_id}, enqueuing for HITL review")
        asyncio.create_task(_enqueue_hitl(request_id, confidence))

    # Make sure it's rounded by only one before returned as response
    score_toxic = round(score_toxic, 1)
    confidence = round(confidence, 1)
    reason = [x | {"score": round(x["score"], 1)} for x in reason]

    return PredictionResponse(
        score_toxic=score_toxic,
        confidence=confidence,
        request_id=request_id,
        reason=[TokenReason(token=x["token"], score=x["score"]) for x in reason],
    )

# root endpoint
@app.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="ok", detail="Rant-Free Model Service running")



