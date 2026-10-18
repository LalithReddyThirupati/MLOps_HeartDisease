# Heart disease risk API - serving image.
# Requires a trained model at models/model.joblib (run `python -m src.train` first).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# Dependencies first so this layer is cached between code changes
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src/ src/
COPY api/ api/
COPY models/model.joblib models/metadata.json models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
