from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import sys

app = Flask(__name__)

# --- Configuración ---
MAX_SEQUENCE_LENGTH = 100
VOCAB_SIZE = 10000

IS_RAILWAY = os.environ.get("RAILWAY_ENVIRONMENT") is not None

if IS_RAILWAY:
    MODEL_PATH = "modelo_sentimiento/classifier.keras"
    VOCABULARY_PATH = "modelo_sentimiento/vocabulary.txt"
else:
    MODEL_PATH = "../modelo_sentimiento/classifier.keras"
    VOCABULARY_PATH = "../modelo_sentimiento/vocabulary.txt"

print(f"Entorno: {'Railway' if IS_RAILWAY else 'Local'}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"VOCABULARY_PATH: {VOCABULARY_PATH}")

# Variables globales
model = None
vectorize_layer = None


def custom_standardization(input_data):
    """Función de estandarización (debe coincidir con el entrenamiento)"""
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, r"[^\w\s]", "")


def load_models():
    """Carga el modelo y reconstruye el vectorizador"""
    global model, vectorize_layer

    try:
        print("=" * 50)
        print("INICIANDO CARGA DE MODELOS")
        print("=" * 50)

        if not os.path.exists(MODEL_PATH):
            print(f"ERROR: No se encuentra {MODEL_PATH}")
            print(f"Archivos en directorio actual:")
            for root, dirs, files in os.walk("."):
                print(f"  {root}:")
                for file in files:
                    print(f"    - {file}")
            return False

        if not os.path.exists(VOCABULARY_PATH):
            print(f"ERROR: No se encuentra {VOCABULARY_PATH}")
            return False

        print(f"Archivos encontrados")

        print("Cargando clasificador...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"Clasificador cargado: {model.count_params()} parámetros")

        print("Cargando vocabulario...")
        with open(VOCABULARY_PATH, "r", encoding="utf-8") as f:
            vocabulary = [line.strip() for line in f]
        print(f"Vocabulario cargado: {len(vocabulary)} palabras")

        print("Reconstruyendo vectorizador...")
        vectorize_layer = layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode="int",
            output_sequence_length=MAX_SEQUENCE_LENGTH,
            standardize=custom_standardization,
        )
        vectorize_layer.set_vocabulary(vocabulary)
        print("Vectorizador reconstruido")

        print("Realizando warmup...")
        test_input = np.array(["test warmup"])
        _ = vectorize_layer(test_input)
        _ = model.predict(np.zeros((1, MAX_SEQUENCE_LENGTH)), verbose=0)
        print("Warmup completado")

        print("=" * 50)
        print("MODELOS LISTOS")
        print("=" * 50)
        return True

    except Exception as e:
        print("=" * 50)
        print("ERROR AL CARGAR MODELOS")
        print("=" * 50)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return False


print("Iniciando carga de modelos...")
models_loaded = load_models()

if not models_loaded:
    print("ADVERTENCIA: Aplicación iniciará sin modelos cargados")
else:
    print("Aplicación lista para recibir peticiones")


@app.route("/", methods=["GET"])
def index():
    """Endpoint raíz"""
    return jsonify(
        {
            "nombre": "API de Análisis de Sentimientos",
            "version": "1.0.0",
            "status": "online" if models_loaded else "degraded",
            "endpoints": {
                "/": "GET - Info de la API",
                "/health": "GET - Estado de salud",
                "/predict": "POST - Predicción de sentimiento",
            },
        }
    )


@app.route("/health", methods=["GET"])
def health():
    """Health check para Railway"""
    is_healthy = model is not None and vectorize_layer is not None

    status_code = 200 if is_healthy else 503

    return (
        jsonify(
            {
                "status": "healthy" if is_healthy else "unhealthy",
                "models_loaded": is_healthy,
                "environment": "railway" if IS_RAILWAY else "local",
            }
        ),
        status_code,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """Predicción de sentimiento"""

    if model is None or vectorize_layer is None:
        return (
            jsonify({"error": "Modelos no cargados. Revisa los logs del servidor."}),
            503,
        )

    try:
        data = request.get_json()

        if not data or "texto" not in data:
            return (
                jsonify(
                    {
                        "error": "Se requiere el campo 'texto'",
                        "ejemplo": {"texto": "This movie was great!"},
                    }
                ),
                400,
            )

        texto = data["texto"].strip()

        if not texto:
            return jsonify({"error": "El texto no puede estar vacío"}), 400

        # Vectorizar
        input_data = np.array([texto])
        texto_vectorizado = vectorize_layer(input_data).numpy()

        # Predecir
        pred = model.predict(texto_vectorizado, verbose=0)[0][0]

        sentimiento = "positivo" if pred >= 0.5 else "negativo"
        confianza = float(pred if pred >= 0.5 else 1 - pred)

        return jsonify(
            {
                "sentimiento": sentimiento,
                "probabilidad_positiva": float(pred),
                "probabilidad_negativa": float(1 - pred),
                "confianza": round(confianza * 100, 2),
                "texto_analizado": texto,
            }
        )

    except Exception as e:
        print(f"Error en predicción: {e}")
        import traceback

        traceback.print_exc()
        return (
            jsonify({"error": "Error al procesar predicción", "details": str(e)}),
            500,
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
