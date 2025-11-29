FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
pip install --no-cache-dir -r requirements.txt

COPY templates/ ./templates/
COPY ./api/ /app
COPY modelo_sentimiento/ ./modelo_sentimiento/

CMD ["python", "app.py"]
