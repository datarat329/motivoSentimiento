
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_HOME=/app

WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/

COPY modelo_sentimiento/ ./modelo_sentimiento/

RUN echo "=== Verificando estructura de archivos ===" && \
    ls -la && \
    echo "=== Contenido de api/ ===" && \
    ls -la api/ && \
    echo "=== Contenido de modelo_sentimiento/ ===" && \
    ls -la modelo_sentimiento/ && \
    echo "=== Verificación completada ==="

EXPOSE 8080

CMD gunicorn --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --timeout 300 \
    --graceful-timeout 300 \
    --keep-alive 5 \
    --log-level info \
    --access-logfile - \
    --error-logfile - \
    --preload \
    api.app:app