# Usa la imagen base de Python oficial (más pequeña)
FROM python:3.11-slim

# Establecer variables de entorno
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app

# Crear el directorio de trabajo
WORKDIR $APP_HOME

# Instalar dependencias del sistema necesarias para TensorFlow (como zlib, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libzbar0 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar los requisitos de Python e instalar las librerías
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código de la API (app.py) y los archivos de configuración
COPY api/ .
# Copiar los modelos entrenados al directorio de trabajo (un nivel arriba)
COPY modelo_sentimiento/ ../modelo_sentimiento/

# Comando para ejecutar la aplicación con Gunicorn (servidor de producción)
# El '0.0.0.0' y el puerto 8000 son estándar para Docker/despliegue
CMD exec gunicorn --bind 0.0.0.0:8000 --workers 1 app:app