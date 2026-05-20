How to prepare environment:
1. (Optional) Prepare your virtual environment.
2. (If not available) Install [PyTorch](https://pytorch.org/get-started/locally/).
3. `pip install -r requirements.txt`

Environment variables to be set:
1. If you need Wandb, `export WANDB_API_KEY=<your api key>` and `export USE_WANDB=true`
2. (other environment variables ... please check `main.py`, it's too much)

How to test:
1. Go to project root (parent of `services`).
2. `docker compose up -d` (or `docker compose build` if you do it for the first time)
3. Go back to `/services/training-service`
4. `python3 main.py`

[Unstable] For checking sample(), do this thing first:
```py
export USE_LOCAL=true
export MONGODB_CONNECTION_STRING="mongodb://localhost:27017"
export MONGODB_DATABASE="feast"
export MONGODB_COLLECTION="feature_history"
```
Then, run `sample.py`.

Note: `pipeline.py` in this service should be consistent with `pipeline.py` in `model-service`.
