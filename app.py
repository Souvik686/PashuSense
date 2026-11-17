"""
PashuSense - High-Speed Production Server
=========================================
Lightning-fast (<0.1s) inference, zero deadlocks, zero external downloads.
"""

import os
import numpy as np
from pathlib import Path

# Reduce TensorFlow log spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
from PIL import Image as PILImage
from breed_data import breed_info

BASE_DIR = Path(__file__).parent
ALLOWED_STATIC_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp"}

# ─── Load Labels ──────────────────────────────────────────────────────────────
with open(BASE_DIR / "labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]
print(f"✅ Labels loaded: {labels}")

# ─── Model Loader (Zero Downloads, In-Memory Warmup) ─────────────────────────
model = None


def get_model():
    """
    Loads model directly from local cattle_breed_model.h5 file.
    Fast, completely offline, zero downloads.
    """
    global model
    if model is not None:
        return model

    model_path = str(BASE_DIR / "cattle_breed_model.h5")

    try:
        print("⏳ Loading model via load_model...")
        model = tf.keras.models.load_model(model_path, compile=False)
        print("✅ Model loaded successfully via load_model!")
    except Exception as err:
        print(f"⚠️ load_model notice: {err}. Rebuilding Sequential architecture...")
        base = tf.keras.applications.MobileNetV2(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )
        base.trainable = False
        m = tf.keras.models.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(5, activation="softmax"),
        ])
        m.load_weights(model_path)
        model = m
        print("✅ Model loaded successfully via Sequential rebuild!")

    # Pre-compile the execution graph with a dummy pass so first user request is instant
    try:
        dummy = np.zeros((1, 224, 224, 3), dtype="float32")
        _ = model(dummy, training=False)
        print("🔥 Model graph compiled and warmed up!")
    except Exception as e:
        print(f"Warmup notice: {e}")

    return model


# Pre-warm model right when server starts
try:
    get_model()
except Exception as e:
    print(f"Initial load note: {e}")

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(BASE_DIR))
CORS(app)


# ── Page Routes ───────────────────────────────────────────────────────────────

@app.route("/")
def login_page():
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None
    }), 200


# ── Prediction API ────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Instant cattle breed prediction (<0.1 second).
    Rejects non-cattle images (humans, dogs, objects) if confidence is low or split.
    """
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]

        # Ensure model is ready
        m = get_model()

        # Read and resize image directly in memory
        img = PILImage.open(file.stream).convert("RGB").resize((224, 224))
        x = np.expand_dims(np.array(img, dtype="float32") / 255.0, axis=0)

        # Ultra-fast eager forward pass (bypasses Keras dataset/generator overhead)
        preds = m(x, training=False).numpy()[0]

        sorted_probs = np.sort(preds)
        top1_prob = float(sorted_probs[-1])
        top2_prob = float(sorted_probs[-2])
        confidence = round(top1_prob * 100, 2)
        idx = int(np.argmax(preds))

        # Margin check: on clear cattle images, the top class dominates (>80% and margin > 35%)
        # On dogs, humans, or non-cattle objects, probabilities are scattered or low.
        margin = top1_prob - top2_prob

        if confidence < 80.0 or margin < 0.35:
            return jsonify({
                "error": f"This does not appear to be a clear cattle image (confidence {confidence}% is low). "
                         "Please upload a clear, close-up photo of a cow."
            }), 200

        breed = labels[idx]
        info = breed_info.get(breed, {"milk_production": "N/A", "health_tips": "N/A"})

        return jsonify({
            "breed": breed,
            "confidence": confidence,
            "milk_production": info["milk_production"],
            "health_tips": info["health_tips"],
        })

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


# ── Static Files Catch-All ───────────────────────────────────────────────────

@app.route("/<path:filename>")
def static_files(filename):
    if Path(filename).suffix.lower() not in ALLOWED_STATIC_EXT:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(BASE_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 PashuSense running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
