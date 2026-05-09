# Rant-Free Project - Model Service
# model.py - Model loading and inference logic
# Author: Versa

import logging
import os
import threading
import time
import random

logger = logging.getLogger("uvicorn.error")


class Model:
    labels = ["toxic"]

    def __init__(self, model_path: str, poll_interval: float = 5.0):
        self._model_path    = model_path
        self._poll_interval = poll_interval
        self._lock          = threading.Lock()
        self._net           = None
        self._version       = "bootstrap"

        # try loading immediately if model already exists
        if model_path == "dummy":
            logger.info("Using dummy model (no file watching)")
            self.predict = self._dummy_predict
        if os.path.exists(model_path):
            self._load(model_path, "bootstrap")

    # --- public interface ---
    @property
    def version(self) -> str:
        with self._lock:
            return self._version

    def predict(self, text: str) -> list[float]:
        """
        Full inference pipeline: text → scores.
        Runs in _gpu_executor thread — safe to block here.
        """
        with self._lock:
            net = self._net

        raise NotImplementedError("Prediction logic not implemented yet")


    def start_watcher(self):
        """Spawn background thread to watch for model file changes."""
        watcher = threading.Thread(
            target=self._watch,
            daemon=True,
        )
        watcher.start()
        logger.info(f"Model watcher started for {self._model_path}")

    # --- internals ---

    def _load(self, path: str, version: str):
        new_net = torch.load(path, map_location="cuda" if torch.cuda.is_available() else "cpu")
        new_net.eval()
        with self._lock:
            self._net     = new_net
            self._version = version
        logger.info(f"Model loaded: {version}")

    def _watch(self):
        last_mtime      = None
        version_counter = 1
        while True:
            try:
                mtime = os.path.getmtime(self._model_path)
                if last_mtime is not None and mtime != last_mtime:
                    self._load(self._model_path, version=f"v{version_counter}")
                    version_counter += 1
                last_mtime = mtime
            except FileNotFoundError:
                pass  # model not written yet, keep waiting
            except Exception as e:
                logger.error(f"Model watcher error: {e}")
            time.sleep(self._poll_interval)

    def _dummy_predict(self, text: str) -> list[float]:
        # This is a placeholder for the actual prediction logic
        rng = random.Random(hash(text))  # Seed with input text for consistent results
        time.sleep(len(text) * 1e-2)
        return [rng.random() for _ in self.labels]
    