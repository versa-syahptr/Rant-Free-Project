import logging
import time

logger = logging.getLogger("uvicorn.error")

def write_prediction(request_id: str, text: str, scores: list[float], confidence: float, model_version: str):
    # Simulate writing to Feast (or any other feature store)
    time.sleep(0.05)  # Simulate network/database latency
    logger.info(f"Writing prediction to Feast [{request_id}]: text='{text[:30]}...', scores={scores}, confidence={confidence:.4f}, model_version={model_version}")
