# Rant-Free Project - Model Service
# model.py - Model loading and inference logic
# Author: Versa and Abdi

from dataclasses import dataclass
import logging
import os
import threading
import time
import random

from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
import torch
from torch import nn

logger = logging.getLogger("uvicorn.error")

def load_embedding_tokenizer_and_model(name_or_path: str):
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    model = AutoModel.from_pretrained(name_or_path, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

@dataclass
class RantFreeModelConfig:
    prefix: str = "Classify:"
    min_toxic_token_score: float = 1.0

class RantFreeClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(RantFreeClassifier, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1, dtype=torch.float16)

    def forward(self, embeddings, attention_mask):
        results = []
        token_sentiments = []
        for emb, attn in zip(embeddings, attention_mask):
            x = self.linear(emb)
            x = x.squeeze(1)
            x = x * attn
            token_sentiments.append(x)
            
            x = x.sum()
            results.append(x)

        h = torch.tensor(results)
        o = 2 * torch.sigmoid(h) - 1
        o = torch.maximum(o, torch.zeros_like(o))

        token_sentiments = torch.stack(token_sentiments)
        return h, o, token_sentiments

class RantFreePipeline:  # Sadly, it can't be inherited from transformers.Pipeline yet.
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            embedding_model: PreTrainedModel,
            classifier_model: RantFreeClassifier,
            config: RantFreeModelConfig | None = None
    ):
        self.embedding_model = embedding_model
        self.classifier_model = classifier_model

        self._tokenizer = tokenizer
        
        if config is None:
            config = RantFreeModelConfig()
        self._prefix = config.prefix
        self._prefix_length = len(self._tokenizer.tokenize(config.prefix))
        self._min_toxic_token_score = config.min_toxic_token_score

        self._device = self.embedding_model.device

    def __call__(self, inputs, *args, **kwargs):
        if args:
            # Should use logger.warning for this, but not now.
            print(f"[WARNING] Ignoring args : {args}")
        
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(**kwargs)
        model_inputs = self.preprocess(inputs, **preprocess_params)
        model_outputs = self.forward(model_inputs, **forward_params)
        outputs = self.postprocess(model_outputs, **postprocess_params)
        return outputs

    def forward(self, model_inputs, **forward_params):
        return self._forward(model_inputs, **forward_params)
    
    def _sanitize_parameters(self, **kwargs):
        return {}, {}, {}  # No parameters for anything
    
    def preprocess(self, inputs):
        texts = [f"{self._prefix} {x}" for x in inputs]
        return self._tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self._device)

    def _forward(self, model_inputs):
        with torch.no_grad():
            outputs = self.embedding_model(**model_inputs)
        
        embeddings = outputs.last_hidden_state[:, self._prefix_length:-1]
        input_ids = model_inputs["input_ids"][:, self._prefix_length:-1]
        attention_mask = model_inputs["attention_mask"][:, self._prefix_length:-1]

        with torch.no_grad():
            h, o, token_sentiments = self.classifier_model(embeddings, attention_mask)
        
        return {
            "raw_scores": h,
            "scores": o,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_sentiments": token_sentiments,
        }

    def postprocess(self, model_outputs):
        old_scores = model_outputs["scores"]
        model_outputs["scores"] = torch.round(old_scores, decimals=1)

        tokens = []
        new_token_sentiments = []
        for x, attn, sentiment in zip(model_outputs["input_ids"], model_outputs["attention_mask"], model_outputs["token_sentiments"].tolist()):
            single_tokens = []
            single_new_token_sentiments = []

            current_token = ""
            current_sentiment = 0.0
            for token, a, s in zip(self._tokenizer.convert_ids_to_tokens(x), attn, sentiment):
                if a == 0 or token == self._tokenizer.pad_token:
                    break

                if token.startswith("Ġ"):
                    if current_token != "" and current_sentiment >= self._min_toxic_token_score:
                        single_tokens.append(current_token)
                        single_new_token_sentiments.append(current_sentiment)
                    
                    current_token = token.lstrip("Ġ")
                    current_sentiment = s
                else:
                    current_token += token
                    current_sentiment += s

            if current_token != "" and current_sentiment >= self._min_toxic_token_score:
                single_tokens.append(current_token)
                single_new_token_sentiments.append(current_sentiment)

            tokens.append(single_tokens)
            new_token_sentiments.append([
                torch.sigmoid(score) for score in single_new_token_sentiments
            ])

        model_outputs["toxic_tokens"] = tokens
        model_outputs["token_sentiments"] = new_token_sentiments
        del model_outputs["input_ids"]
        del model_outputs["attention_mask"]
        
        return model_outputs

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
