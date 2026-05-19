# Rant-Free Project - Model Service
# model.py - Model loading and inference logic
# Author: Versa and Abdi

from dataclasses import dataclass
import logging
import os
import threading
import time
import random

import torch
from transformers import PreTrainedModel

from pipeline import RantFreeClassifier, RantFreeModelConfig, RantFreePipeline, load_embedding_tokenizer_and_model

logger = logging.getLogger("uvicorn.error")

class RantFreeModel:
    def __init__(
            self,
            embedding_model_name_or_path: str,
            classifier_model_path: str,
            config: RantFreeModelConfig | None = None,
            poll_interval: float = 5.0
    ):
        self._embedding_model_name_or_path = embedding_model_name_or_path
        self._classifier_model_path = classifier_model_path
        self._config = config
        self._poll_interval = poll_interval
        self._lock = threading.Lock()
        self._pipeline = None
        self._version = "bootstrap"

        # try loading immediately if model already exists
        if embedding_model_name_or_path == "dummy" and classifier_model_path == "dummy":
            logger.info("Using dummy model (no file watching)")
            self.predict = self._dummy_predict
        else:
            self._load(embedding_model_name_or_path, classifier_model_path, config, "bootstrap")
    
    # --- public interface ---
    @property
    def version(self) -> str:
        with self._lock:
            return self._version

    def predict(self, text: str) -> tuple[float, list[dict]]:
        """
        Full inference pipeline: text → scores.
        Runs in _gpu_executor thread — safe to block here.
        """
        with self._lock:
            pipeline = self._pipeline
        
        assert pipeline is not None
        with torch.no_grad():
            result = pipeline([text])
        score_toxic = float(result["scores"][0])
        reason = [
            {"token": token, "score": float(score)}
            for token, score in zip(result["toxic_tokens"][0], result["token_sentiments"][0])
        ]
        return score_toxic, reason
    
    def start_watcher(self):
        """Spawn background thread to watch for model file changes."""
        # Watching embedding model is not implemented yet since it can be from external.
        self.start_classifier_watcher()

    def start_classifier_watcher(self):
        """Spawn background thread to watch for classifier model file changes."""
        watcher = threading.Thread(
            target=self._watch_classifier,
            daemon=True,
        )
        watcher.start()
        logger.info(f"Classifier model watcher started for {self._classifier_model_path}")

    # --- internals ---

    def _load(
            self,
            embedding_name_or_path: str,
            classifier_path: str,
            config: RantFreeModelConfig | None,
            version: str
    ):
        # I wonder ... why those arguments can't be inferred in self?
        tokenizer, embedding_model = self._load_embedding(embedding_name_or_path)
        classifier_model = self._load_classifier(classifier_path, embedding_model)
        pipeline = RantFreePipeline(
            tokenizer=tokenizer,
            embedding_model=embedding_model,
            classifier_model=classifier_model,
            config=config,
        )
        pipeline.eval()

        with self._lock:
            self._pipeline = pipeline
            self._version = version
        
        logger.info(f"Pipeline loaded: {version}")

    def _load_embedding(self, name_or_path: str):
        tokenizer, model = load_embedding_tokenizer_and_model(name_or_path)
        model.eval()

        logger.info(f"Embedding model loaded")
        return tokenizer, model
    
    def _load_classifier(self, path: str, embedding_model: PreTrainedModel):
        embedding_dim = embedding_model.get_input_embeddings().embedding_dim
        model = RantFreeClassifier(embedding_dim=embedding_dim)
        model.load_state_dict(torch.load(path, weights_only=True))
        model = model.to(embedding_model.device)
        model.eval()
        logger.info(f"Classifier model loaded (embedding_dim: {embedding_dim})")
        return model
    
    def _watch_classifier(self):
        last_mtime = None
        version_counter = 1
        while True:
            try:
                
                mtime = os.path.getmtime(self._classifier_model_path)
                if last_mtime is not None and mtime != last_mtime:
                    self._load(
                        embedding_name_or_path=self._embedding_model_name_or_path,
                        classifier_path=self._classifier_model_path,
                        config=self._config,
                        version=f"v{version_counter}",
                    )
                    version_counter += 1
                last_mtime = mtime
            except FileNotFoundError:
                pass  # model not written yet, keep waiting
            except Exception as e:
                logger.error(f"Classifier model watcher error: {e}")
            time.sleep(self._poll_interval)

    def _dummy_predict(self, text: str) -> tuple[float, list[dict]]:
        # This is a placeholder for the actual prediction logic
        rng = random.Random(hash(text))  # Seed with input text for consistent results
        time.sleep(len(text) * 1e-2)
        return rng.random(), []
