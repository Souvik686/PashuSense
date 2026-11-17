import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np
from PIL import Image
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

print("TensorFlow Version:", tf.__version__)

# 1. Build Exact Model Architecture
base = tf.keras.applications.MobileNetV2(
    weights=None, include_top=False, input_shape=(224, 224, 3)
)
base.trainable = False

model = tf.keras.models.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(5, activation="softmax"),
])

# Pass dummy through model so layer input shapes are fully resolved
dummy = np.zeros((1, 224, 224, 3), dtype="float32")
_ = model(dummy, training=False)

# 2. Load Weights from cattle_breed_model.h5
h5_path = "cattle_breed_model.h5"
model.load_weights(h5_path)
print("Successfully loaded weights from cattle_breed_model.h5!")

# 3. Test on Cattles/download.jpeg
img = Image.open("Cattles/download.jpeg").convert("RGB").resize((224, 224))
x = np.expand_dims(np.array(img, dtype="float32") / 255.0, axis=0)

with open("labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

preds_h5 = model(x, training=False).numpy()[0]
top_idx = int(np.argmax(preds_h5))
print(f"H5 Model Prediction: {labels[top_idx]} ({round(float(preds_h5[top_idx]) * 100, 2)}%)")
print("All H5 probabilities:", {labels[i]: round(float(preds_h5[i]), 4) for i in range(len(labels))})

# 4. Convert to TFLite
print("\nConverting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = "cattle_breed_model.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"Saved {tflite_path} ({len(tflite_model)} bytes / {round(len(tflite_model)/1024/1024, 2)} MB)!")

# 5. Verify TFLite model on the same image
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], x)
interpreter.invoke()
preds_tflite = interpreter.get_tensor(output_details[0]["index"])[0]

top_idx_tflite = int(np.argmax(preds_tflite))
print(f"\nTFLite Model Prediction: {labels[top_idx_tflite]} ({round(float(preds_tflite[top_idx_tflite]) * 100, 2)}%)")
print("All TFLite probabilities:", {labels[i]: round(float(preds_tflite[i]), 4) for i in range(len(labels))})
