# services/model-service/Dockerfile
# Rant-Free Project - Model Service
# Build context: ./services
#
# Local (CPU):
#   docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
#     -f services/model-service/Dockerfile -t rant-free-model-service:dev ./services
#
# Production (CUDA 11.8):
#   docker build \
#     -f services/model-service/Dockerfile -t rant-free-model-service:latest ./services

FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu118

WORKDIR /app

COPY model-service/requirements.txt .
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
RUN pip install --no-cache-dir -r requirements.txt

COPY feature_store/ /app/feature_store/
RUN pip install --no-cache-dir -e /app/feature_store/ 
COPY model-service/ .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]