from flask import Flask, request, jsonify
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
logger.info("Importando TensorFlow...")
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

logger.info(f"TensorFlow {tf.__version__} importado exitosamente")

app = Flask(__name__)

MAX_SEQUENCE_LENGTH = 100
VOCAB_SIZE = 10000

MODEL_PATH = "modelo_sentimiento/classifier.keras"
VOCABULARY_PATH = "modelo_sentimiento/vocabulary.txt"

logger.info("=" * 60)
logger.info("INICIANDO APLICACIÓN EN DOCKER")
logger.info("=" * 60)
logger.info(f"Directorio de trabajo: {os.getcwd()}")
logger.info(f"MODEL_PATH: {MODEL_PATH}")
logger.info(f"VOCABULARY_PATH: {VOCABULARY_PATH}")
logger.info(f"Python: {sys.version.split()[0]}")
logger.info(f"TensorFlow: {tf.__version__}")
logger.info("=" * 60)

model = None
vectorize_layer = None
models_loading = True


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, r"[^\w\s]", "")


def load_models():
    global model, vectorize_layer, models_loading

    try:
        logger.info("Iniciando carga de modelos...")

        logger.info("Contenido del directorio actual:")
        for item in os.listdir("."):
            logger.info(f"  - {item}")

        logger.info(f"Verificando {MODEL_PATH}...")
        if not os.path.exists(MODEL_PATH):
            logger.error(f"NO ENCONTRADO: {MODEL_PATH}")
            models_loading = False
            return False
        logger.info("Archivo de modelo encontrado")

        logger.info(f"Verificando {VOCABULARY_PATH}...")
        if not os.path.exists(VOCABULARY_PATH):
            logger.error(f"NO ENCONTRADO: {VOCABULARY_PATH}")
            models_loading = False
            return False
        logger.info("Archivo de vocabulario encontrado")

        logger.info("Cargando modelo clasificador...")

        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e1:
            logger.warning(f"Método 1 falló: {e1}")
            try:
                logger.info("Intentando método alternativo (compile=False)...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                model.compile(
                    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
                )
            except Exception as e2:
                logger.error(f"Método 2 falló: {e2}")
                logger.info("Intentando método alternativo (reconstruir modelo)...")
                model = models.Sequential(
                    [
                        layers.Embedding(
                            input_dim=VOCAB_SIZE,
                            output_dim=16,
                            input_length=MAX_SEQUENCE_LENGTH,
                        ),
                        layers.GlobalAveragePooling1D(),
                        layers.Dense(16, activation="relu"),
                        layers.Dense(1, activation="sigmoid"),
                    ]
                )
                model.compile(
                    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
                )

                weights_path = MODEL_PATH.replace(".keras", "_weights.h5")
                if os.path.exists(weights_path):
                    model.load_weights(weights_path)
                else:
                    raise Exception("No se pudo cargar el modelo con ningún método")

        logger.info(f"Clasificador cargado: {model.count_params():,} parámetros")

        logger.info("Cargando vocabulario...")
        with open(VOCABULARY_PATH, "r", encoding="utf-8") as f:
            vocabulary = [line.strip() for line in f]
        logger.info(f"Vocabulario cargado: {len(vocabulary):,} palabras")

        logger.info("Reconstruyendo vectorizador...")
        vectorize_layer = layers.TextVectorization(
            max_tokens=VOCAB_SIZE,
            output_mode="int",
            output_sequence_length=MAX_SEQUENCE_LENGTH,
            standardize=custom_standardization,
        )
        vectorize_layer.set_vocabulary(vocabulary)
        logger.info("Vectorizador reconstruido")

        logger.info("Realizando warmup del modelo...")
        test_input = np.array(["test"])
        _ = vectorize_layer(test_input)
        _ = model.predict(np.zeros((1, MAX_SEQUENCE_LENGTH)), verbose=0)
        logger.info("Warmup completado")

        models_loading = False
        logger.info("=" * 60)
        logger.info("TODOS LOS MODELOS CARGADOS EXITOSAMENTE")
        logger.info("=" * 60)
        return True

    except Exception as e:
        models_loading = False
        logger.error("=" * 60)
        logger.error("ERROR AL CARGAR MODELOS")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False


import threading


def load_models_async():
    global models_loaded
    models_loaded = load_models()
    if models_loaded:
        logger.info("Aplicación lista para recibir peticiones")
    else:
        logger.warning("Modelos no pudieron cargarse - /predict no funcionará")


logger.info("Iniciando carga de modelos en background...")
loading_thread = threading.Thread(target=load_models_async, daemon=True)
loading_thread.start()


@app.route("/", methods=["GET"])
def index():
    return (
        jsonify(
            {
                "service": "API de Análisis de Sentimientos",
                "version": "1.0.0",
                "status": "online" if models_loaded else "degraded",
                "models_loaded": models_loaded,
                "endpoints": {
                    "/": "GET - Información de la API",
                    "/health": "GET - Estado de salud",
                    "/predict": "POST - Predicción de sentimiento",
                },
                "ejemplo": {
                    "endpoint": "/predict",
                    "method": "POST",
                    "body": {"texto": "This movie was absolutely fantastic!"},
                },
            }
        ),
        200,
    )


@app.route("/health", methods=["GET"])
def health():
    is_ready = model is not None and vectorize_layer is not None

    if models_loading:
        status = "loading"
    elif is_ready:
        status = "ready"
    else:
        status = "error"

    response = {
        "status": status,
        "models_loaded": is_ready,
        "tensorflow_version": tf.__version__,
        "python_version": sys.version.split()[0],
    }

    return jsonify(response), 200


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorize_layer is None:
        return (
            jsonify(
                {
                    "error": "Modelos no disponibles",
                    "details": "Los modelos no se cargaron correctamente. Revisa los logs.",
                }
            ),
            503,
        )

    try:
        data = request.get_json()

        if not data:
            return (
                jsonify({"error": "Se esperaba un JSON en el cuerpo de la petición"}),
                400,
            )

        if "texto" not in data:
            return (
                jsonify(
                    {
                        "error": "Campo 'texto' requerido",
                        "ejemplo": {"texto": "This movie was great!"},
                    }
                ),
                400,
            )

        texto = data["texto"].strip()

        if not texto:
            return jsonify({"error": "El texto no puede estar vacío"}), 400

        if len(texto) > 5000:
            return (
                jsonify(
                    {"error": "El texto es demasiado largo (máximo 5000 caracteres)"}
                ),
                400,
            )

        input_data = np.array([texto])
        texto_vectorizado = vectorize_layer(input_data).numpy()

        pred = model.predict(texto_vectorizado, verbose=0)[0][0]

        sentimiento = "positivo" if pred >= 0.5 else "negativo"
        confianza = float(pred if pred >= 0.5 else 1 - pred)

        return (
            jsonify(
                {
                    "sentimiento": sentimiento,
                    "probabilidad_positiva": round(float(pred), 4),
                    "probabilidad_negativa": round(float(1 - pred), 4),
                    "confianza": round(confianza * 100, 2),
                    "texto_analizado": texto,
                }
            ),
            200,
        )

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        import traceback

        logger.error(traceback.format_exc())

        return (
            jsonify({"error": "Error al procesar la predicción", "details": str(e)}),
            500,
        )


if __name__ == "__main__":
    port = 8000
    logger.info(f"Iniciando servidor en puerto {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
