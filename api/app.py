from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os

app = Flask(__name__)

# --- Rutas de Archivos ---
MODEL_PATH = "../modelo_sentimiento/classifier.keras"
VOCABULARY_PATH = "../modelo_sentimiento/vocabulary.txt"

# --- Configuración (debe coincidir con el entrenamiento) ---
MAX_SEQUENCE_LENGTH = 100
VOCAB_SIZE = 10000

# -------------------------------
# Cargar modelo y reconstruir vectorizador
# -------------------------------
model = None
vectorize_layer = None


def custom_standardization(input_data):
    """Misma función de estandarización usada en el entrenamiento"""
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, r"[^\w\s]", "")


def load_models():
    """Carga el modelo y reconstruye el vectorizador desde el vocabulario"""
    global model, vectorize_layer

    try:
        print("Cargando modelo clasificador...")
        model = tf.keras.models.load_model(MODEL_PATH)

        print("Reconstruyendo vectorizador desde vocabulario...")

        # Leer el vocabulario guardado
        with open(VOCABULARY_PATH, "r", encoding="utf-8") as f:
            vocabulary = [line.strip() for line in f]

        print(f"Vocabulario cargado: {len(vocabulary)} palabras")

        # Recrear la capa TextVectorization con el vocabulario cargado
        vectorize_layer = layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode="int",
            output_sequence_length=MAX_SEQUENCE_LENGTH,
            standardize=custom_standardization,
        )

        # Establecer el vocabulario (sin necesidad de adapt)
        vectorize_layer.set_vocabulary(vocabulary)

        # Pre-compilación para optimizar primera predicción
        test_input = np.array(["test"])
        _ = vectorize_layer(test_input)
        _ = model.predict(np.zeros((1, MAX_SEQUENCE_LENGTH)), verbose=0)

        print("✓ Modelos cargados y listos.")
        return True

    except FileNotFoundError as e:
        print(f"❌ Error: Archivo no encontrado - {e}")
        print("Asegúrate de que los archivos existen en:")
        print(f"  - {MODEL_PATH}")
        print(f"  - {VOCABULARY_PATH}")
        return False
    except Exception as e:
        print(f"❌ Error al cargar los modelos: {e}")
        import traceback

        traceback.print_exc()
        return False


# Cargar modelos al iniciar la aplicación
if not load_models():
    print("⚠️  ADVERTENCIA: La API se iniciará pero las predicciones fallarán")


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint para predecir el sentimiento de una reseña."""

    # Verificar que los modelos estén cargados
    if model is None or vectorize_layer is None:
        return (
            jsonify(
                {
                    "error": "Los modelos no están cargados correctamente.",
                    "details": "Revisa los logs del servidor para más información.",
                }
            ),
            503,
        )  # Service Unavailable

    try:
        # Validar entrada
        data = request.get_json()
        if not data:
            return (
                jsonify({"error": "Se esperaba un JSON en el cuerpo de la petición."}),
                400,
            )

        if "texto" not in data:
            return (
                jsonify(
                    {
                        "error": "Falta el campo 'texto' en el JSON de entrada.",
                        "ejemplo": {"texto": "Esta película es excelente"},
                    }
                ),
                400,
            )

        texto = data["texto"]

        # Validar que el texto no esté vacío
        if not texto or not texto.strip():
            return jsonify({"error": "El campo 'texto' no puede estar vacío."}), 400

        # Preparar entrada
        input_data = np.array([texto])

        # 1. Vectorizar el texto (convierte string a secuencia de enteros)
        texto_vectorizado = vectorize_layer(input_data).numpy()

        # 2. Predicción
        pred = model.predict(texto_vectorizado, verbose=0)[0][0]

        # 3. Determinar sentimiento
        sentimiento = "positivo" if pred >= 0.5 else "negativo"

        # 4. Calcular nivel de confianza
        confianza = float(pred if pred >= 0.5 else 1 - pred)

        # Formatear la respuesta
        return (
            jsonify(
                {
                    "sentimiento": sentimiento,
                    "probabilidad_positiva": float(pred),
                    "probabilidad_negativa": float(1 - pred),
                    "confianza": round(confianza * 100, 2),
                    "texto_analizado": texto,
                }
            ),
            200,
        )

    except Exception as e:
        print(f"Error en predicción: {e}")
        import traceback

        traceback.print_exc()
        return (
            jsonify(
                {
                    "error": "Ocurrió un error al procesar la predicción.",
                    "details": str(e),
                }
            ),
            500,
        )


@app.route("/health", methods=["GET"])
def health():
    """Endpoint de salud para verificar que la API está funcionando."""
    models_loaded = model is not None and vectorize_layer is not None

    return jsonify(
        {
            "status": "ok" if models_loaded else "degraded",
            "message": "API de Sentimiento funcionando.",
            "models_loaded": models_loaded,
            "endpoints": {
                "/health": "GET - Verificar estado de la API",
                "/predict": "POST - Predecir sentimiento de texto",
            },
        }
    ), (200 if models_loaded else 503)


@app.route("/", methods=["GET"])
def index():
    """Endpoint raíz con información de la API."""
    return (
        jsonify(
            {
                "nombre": "API de Análisis de Sentimientos",
                "version": "1.0.0",
                "descripcion": "Analiza el sentimiento de textos en inglés (positivo/negativo)",
                "ejemplo_uso": {
                    "endpoint": "/predict",
                    "metodo": "POST",
                    "body": {"texto": "This movie was absolutely fantastic!"},
                },
            }
        ),
        200,
    )


if __name__ == "__main__":
    # Para desarrollo local
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
