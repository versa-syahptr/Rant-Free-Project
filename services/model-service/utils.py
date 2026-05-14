import os

def get_info_from_env():
    embedding_model_name_or_path = os.getenv("EMBEDDING_MODEL_NAME_OR_PATH")
    if embedding_model_name_or_path is None:
        raise ValueError("EMBEDDING_MODEL_NAME_OR_PATH environment variables should be set.")

    classifier_model_path = os.getenv("CLASSIFIER_MODEL_PATH")
    if classifier_model_path is None:
        raise ValueError("CLASSIFIER_MODEL_PATH environment variables should be set.")
    
    return embedding_model_name_or_path, classifier_model_path
