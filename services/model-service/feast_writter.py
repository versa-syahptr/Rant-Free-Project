import logging
import time
import pandas as pd
from datetime import datetime, timezone

# import from services/feature_store/store.py
# install with: `pip install -e services/feature_store` from repo root if run locally,
# if use docker, the dependency is already included in the image so no need to install separately. Just make sure to import correctly.
from feature_store.store import get_store

logger = logging.getLogger("uvicorn.error")

store = get_store() # feast.FeatureStore

def write_prediction(
        request_id: str,
        text: str,
        scores: list[float],
        confidence: float,
        language: str,
        model_version: str,
):
    # DEPRECATED, please use v2!
    # Simulate writing to Feast (or any other feature store)
    # versa note: i don't know why we need 2 functions for this task but better to not touch this function although it's not used anywhere else
    time.sleep(0.05)  # Simulate network/database latency
    logger.info(f"Writing prediction to Feast [{request_id}]: text='{text[:30]}...', scores={scores}, confidence={confidence:.4f}, language={language}, model_version={model_version}")

def write_prediction_v2(
        request_id: str,
        text: str,
        score_toxic: float,
        confidence: float,
        language: str,
        model_version: str,
):
    toxic = score_toxic >= 0.5

    df = pd.DataFrame({
        "request_id":      [request_id],
        "event_timestamp": [pd.Timestamp(datetime.now(timezone.utc))],
        "text":            [text],
        "language":        [language],
        "score_toxic":     [score_toxic],
        "confidence":      [confidence],
        "model_version":   [model_version],
        "toxic":           [toxic],
    })

    # actually write to feast
    store.write_to_offline_store(
        feature_view_name="comment_features",
        df=df,
    )

    logger.info(f"Writing prediction to Feast [{request_id}]: text='{text[:30]}...', score_toxic={score_toxic}, confidence={confidence:.4f}, language={language}, model_version={model_version}, toxic={toxic}")
