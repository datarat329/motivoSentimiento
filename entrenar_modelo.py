import sys
import os
import tensorflow as tf

# --- Configuración de codificación ANTES de cualquier otra cosa ---
os.environ["PYTHONIOENCODING"] = "utf-8"
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
import numpy as np

# --- Configuración ---
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 16
OUTPUT_DIR = "modelo_sentimiento"

os.makedirs(OUTPUT_DIR, exist_ok=True)

## -------------------------------
## 1. Cargar y Preparar Dataset IMDB
## -------------------------------
print("1. Cargando y preparando el dataset IMDB...")
(x_train_int, y_train), (x_test_int, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

word_index = imdb.get_word_index()
index_to_word = {v + 3: k for k, v in word_index.items()}
index_to_word[0] = "[PAD]"
index_to_word[1] = "[START]"
index_to_word[2] = "[UNK]"


def decode_review(int_sequence):
    return " ".join([index_to_word.get(i, "[UNK]") for i in int_sequence])


x_train_text = np.array([decode_review(s) for s in x_train_int])
x_test_text = np.array([decode_review(s) for s in x_test_int])
y_train = np.array(y_train)

## -------------------------------
## 2. Capa de TextVectorization (Preprocesamiento)
## -------------------------------
print("2. Creando la capa TextVectorization...")


# SOLUCIÓN 1: Usar standardize personalizado para limpiar caracteres problemáticos
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    # Eliminar caracteres especiales problemáticos
    return tf.strings.regex_replace(lowercase, r"[^\w\s]", "")


vectorize_layer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_SEQUENCE_LENGTH,
    standardize=custom_standardization,  # Añadir estandarización personalizada
)

vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(x_train_text))

## -------------------------------
## 3. Construir el Modelo Clasificador
## -------------------------------
print("3. Construyendo y compilando el modelo...")
model = models.Sequential(
    [
        layers.Input(shape=(MAX_SEQUENCE_LENGTH,)),
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

## -------------------------------
## 4. Entrenamiento
## -------------------------------
print("4. Iniciando entrenamiento (3 épocas)...")
x_train_sequences = vectorize_layer(x_train_text).numpy()

model.fit(
    x_train_sequences, y_train, epochs=3, batch_size=32, validation_split=0.2, verbose=1
)

## -------------------------------
## 5. Guardar Modelos
## -------------------------------
print("5. Guardando modelos...")

try:
    # SOLUCIÓN 2: Guardar en formato TensorFlow SavedModel (más robusto)
    vectorizer_model = tf.keras.Sequential(
        [tf.keras.Input(shape=(1,), dtype=tf.string), vectorize_layer]
    )

    # Usar formato SavedModel en lugar de .keras
    vectorizer_model.save(f"{OUTPUT_DIR}/vectorizer", save_format="tf")
    print("✓ Vectorizador guardado exitosamente en formato SavedModel")

    # Guardar el clasificador
    model.save(f"{OUTPUT_DIR}/classifier.keras")
    print("✓ Clasificador guardado exitosamente")

except Exception as e:
    print(f"Error al guardar: {e}")

    # SOLUCIÓN 3 ALTERNATIVA: Guardar solo los pesos y configuración
    print("\nIntentando método alternativo...")
    try:
        # Guardar configuración del vectorizador
        vocab = vectorize_layer.get_vocabulary()
        with open(f"{OUTPUT_DIR}/vocabulary.txt", "w", encoding="utf-8") as f:
            for word in vocab:
                f.write(word + "\n")

        # Guardar el modelo clasificador
        model.save(f"{OUTPUT_DIR}/classifier.keras")

        print("Vocabulario y clasificador guardados con método alternativo")
        print(f"  - Vocabulario: {OUTPUT_DIR}/vocabulary.txt")
        print(f"  - Clasificador: {OUTPUT_DIR}/classifier.keras")

    except Exception as e2:
        print(f"Error en método alternativo: {e2}")

## -------------------------------
## 6. Evaluar en datos de prueba
## -------------------------------
print("\n6. Evaluando en datos de prueba...")
x_test_sequences = vectorize_layer(x_test_text).numpy()
test_loss, test_acc = model.evaluate(x_test_sequences, y_test, verbose=0)
print(f"Precisión en test: {test_acc:.4f}")

print("\n✓ Proceso completado exitosamente.")
