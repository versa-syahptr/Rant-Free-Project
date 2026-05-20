How to prepare environment:
1. (Optional) Prepare your virtual environment.
2. (If not available) Install [PyTorch](https://pytorch.org/get-started/locally/).
3. pip install -r requirements.txt

Environment variables to be set:
1. `EMBEDDING_MODEL_NAME_OR_PATH`
2. `CLASSIFIER_MODEL_PATH`
3. `USE_DUMMY` (set `"true"` for activate)

Example (in terminal):
```bash
export EMBEDDING_MODEL_NAME_OR_PATH="BAAI/bge-small-en"
export CLASSIFIER_MODEL_PATH="data/model_weights.pth"
export USE_DUMMY="false"
```

How to init model if not exists yet:
1. Run init_untrained_classifier.py.

How to (actually) test:
1. uvicorn main:app --reload
2. curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text":"bro made a statement so trash"}'

How to unit test:
1. pytest test_model.py
