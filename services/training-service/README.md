How to prepare environment:
1. (Optional) Prepare your virtual environment.
2. (If not available) Install [PyTorch](https://pytorch.org/get-started/locally/).
3. `pip install -r requirements.txt`

Environment variables to be set:
1. (please check `main.py`, it's too much)
2. (optional) `WANDB_API_KEY` if `USE_WANDB=true`

How to test: `python3 main.py`

Note: `pipeline.py` in this service should be consistent with `pipeline.py` in `model-service`.
