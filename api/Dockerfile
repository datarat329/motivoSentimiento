FROM python:3.11-slim

ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
    
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ .
COPY modelo_sentimiento/ modelo_sentimiento/

CMD exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 1 app:app