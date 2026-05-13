# How to run: dotenv run -- python3 init_untrained_classifier.py

import os
from pathlib import Path
import time

import torch
from transformers import AutoModel, AutoTokenizer

from model import RantFreeClassifier

if __name__ == "__main__":
    embedding_model_name_or_path = os.getenv("EMBEDDING_MODEL_NAME_OR_PATH")
    if embedding_model_name_or_path is None:
        raise ValueError("EMBEDDING_MODEL_NAME_OR_PATH environment variables should be set.")

    classifier_model_path = os.getenv("CLASSIFIER_MODEL_PATH")
    if classifier_model_path is None:
        raise ValueError("CLASSIFIER_MODEL_PATH environment variables should be set.")
    
    print("Initializing ....")
    start_time = time.time()
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name_or_path)
    embedding_model = AutoModel.from_pretrained(embedding_model_name_or_path, torch_dtype=torch.float16, device_map="auto")
    embedding_dim = embedding_model.get_input_embeddings().embedding_dim
    classifier_model = RantFreeClassifier(embedding_dim=embedding_dim)

    classifier_model_path = Path(classifier_model_path)
    classifier_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier_model.state_dict(), classifier_model_path)

    duration = time.time() - start_time
    print(f"Untrained classifier has been initialized. ({duration:.2f} s)")
