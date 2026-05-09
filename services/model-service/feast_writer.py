import os
from pathlib import Path
import logging
import time

from feast import FeatureStore
import pandas as pd

logger = logging.getLogger("uvicorn.error")

class FeastWriter:
    def __init__(self, *, use_feature_store: bool = True):
        if use_feature_store:
            repo_path = os.getenv("FEATURE_STORE_REPO_PATH")
            fs_yaml_file = os.getenv("FEATURE_STORE_YAML", "../feature-store/feature_repo/feature_store.yaml")
            fs_yaml_file = Path(fs_yaml_file)

            self.feature_store = FeatureStore(
                repo_path=repo_path,
                fs_yaml_file=fs_yaml_file,
            )
            
        else:
            self.feature_store = None

    def write_prediction(
            self,
            request_id: str,
            text: str,
            language: str,
            score_toxic: float,
            confidence: float,
            model_version: str,
            *,
            use_feature_store: bool = True
    ):
        if self.feature_store is not None:
            rows = [{
                "request_id": request_id,
                "text": text,
                "language": language,
                "score_toxic": score_toxic,
                "confidence": confidence,
                "model_version": model_version,
                "toxic": None,
                "created_timestamp": pd.Timestamp.now()
            }]
            df = pd.DataFrame(rows)
            self.feature_store.push("predictions", df)
        else:
            time.sleep(0.05)  # Simulate network/database latency
        
        logger.info(f"Writing prediction to Feast [{request_id}]: text='{text[:30]}...', language={language}, score_toxic={score_toxic}, confidence={confidence:.4f}, model_version={model_version}, use_feature_store={use_feature_store}")
