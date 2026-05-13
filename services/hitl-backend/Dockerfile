FROM python:3.12-slim

WORKDIR /app

COPY hitl-backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_store/ /app/feature_store/
RUN pip install --no-cache-dir -e /app/feature_store/ 
COPY hitl-backend/ .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
