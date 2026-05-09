How to prepare:
1. (Optional) Prepare your virtual environment.
2. pip install -r requirements.txt

How to test:
1. uvicorn main:app --reload
2. curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"text":"bro made a statement so trash"}'
