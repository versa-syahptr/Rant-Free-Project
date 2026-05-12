import logging
import time

logger = logging.getLogger("uvicorn.error")

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
    toxic = None

    # Simulate writing to Feast (or any other feature store)
    time.sleep(0.05)  # Simulate network/database latency
    logger.info(f"Writing prediction to Feast [{request_id}]: text='{text[:30]}...', score_toxic={score_toxic}, confidence={confidence:.4f}, language={language}, model_version={model_version}, toxic={toxic}")
