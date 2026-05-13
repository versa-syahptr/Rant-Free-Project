from pathlib import Path
import time

import torch

from model import RantFreeClassifier, load_embedding_tokenizer_and_model
from utils import get_info_from_env

if __name__ == "__main__":
    embedding_model_name_or_path, classifier_model_path = get_info_from_env()
    
    print("Initializing ....")
    start_time = time.time()
    embedding_tokenizer, embedding_model = load_embedding_tokenizer_and_model(
        embedding_model_name_or_path
    )
    embedding_dim = embedding_model.get_input_embeddings().embedding_dim
    classifier_model = RantFreeClassifier(embedding_dim=embedding_dim)

    classifier_model_path = Path(classifier_model_path)
    classifier_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(classifier_model.state_dict(), classifier_model_path)

    duration = time.time() - start_time
    print(f"Untrained classifier has been initialized. ({duration:.2f} s)")
