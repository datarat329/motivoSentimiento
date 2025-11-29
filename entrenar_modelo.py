import sys
import os
import tensorflow as tf

os.environ["PYTHONIOENENCODING"] = "utf-8"

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
import numpy as np

print(f"TensorFlow version: {tf.__version__}")

VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 16
OUTPUT_DIR = "modelo_sentimiento"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

print("2. Creando la capa TextVectorization...")


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, r"[^\w\s]", "")


vectorize_layer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=MAX_SEQUENCE_LENGTH,
    standardize=custom_standardization,
)
vectorize_layer.adapt(tf.data.Dataset.from_tensor_slices(x_train_text))

print("3. Construyendo y compilando el modelo...")
model = models.Sequential(
    [
        layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH,
        ),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print("4. Iniciando entrenamiento (3 épocas)...")
x_train_sequences = vectorize_layer(x_train_text).numpy()
model.fit(
    x_train_sequences,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
)

print("5. Guardando modelos...")
try:
    vocab = vectorize_layer.get_vocabulary()
    vocab_path = f"{OUTPUT_DIR}/vocabulary.txt"
    with open(vocab_path, "w", encoding="utf-8") as f:
        for word in vocab:
            f.write(word + "\n")
    print(f"Vocabulario guardado: {vocab_path}")

    classifier_path = f"{OUTPUT_DIR}/classifier.keras"
    model.save(classifier_path, save_format="keras_v3")
    print(f"Clasificador guardado: {classifier_path}")

except Exception as e:
    print(f"Error al guardar: {e}")
    import traceback

    traceback.print_exc()

print("\n6. Evaluando en datos de prueba...")
x_test_sequences = vectorize_layer(x_test_text).numpy()
test_loss, test_acc = model.evaluate(x_test_sequences, y_test, verbose=0)
print(f"Precisión en test: {test_acc:.4f}")

print("\n7. Probando predicción...")
test_texts = [
    "This movie was absolutely fantastic!",
    "Terrible waste of time, do not watch.",
]
for text in test_texts:
    text_vec = vectorize_layer(np.array([text])).numpy()
    pred = model.predict(text_vec, verbose=0)[0][0]
    sentiment = "positivo" if pred >= 0.5 else "negativo"
    print(f"Texto: '{text}'")
    print(f"Predicción: {sentiment} (confianza: {pred:.4f})\n")

print("Proceso completado exitosamente.")
