"""
PashuSense - High-Speed Production Server
=========================================
Optimized for instant responses and zero 502 errors on cloud instances (Render, etc.)
"""

import os
import threading
import numpy as np
from pathlib import Path

# Reduce TensorFlow log spam & thread contention on low-core instances
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# 1 thread for CPU inference prevents thread deadlocks & CPU throttling on cloud
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from PIL import Image as PILImage

from breed_data import breed_info

BASE_DIR = Path(__file__).parent
ALLOWED_STATIC_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp"}

# Confidence Threshold: Real cows score 85%+ on their breed class.
MIN_CONFIDENCE = 85.0

# ─── Load Labels ──────────────────────────────────────────────────────────────
with open(BASE_DIR / "labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]
print(f"✅ Labels: {labels}")

# ─── Safe Model Loader with Background Warmup ─────────────────────────────────
breed_model = None
_model_lock = threading.Lock()


def load_breed_model():
    """
    Safely loads cattle_breed_model.h5 from local repository.
    Includes fallback if weights=None has layer shape variance.
    """
    global breed_model
    with _model_lock:
        if breed_model is not None:
            return breed_model

        try:
            print("⏳ Loading cattle breed model from local weights...")
            _base = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
            _base.trainable = False

            m = models.Sequential([
                _base,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(0.3),
                layers.Dense(5, activation="softmax"),
            ])
            m.load_weights(str(BASE_DIR / "cattle_breed_model.h5"))
            m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
            breed_model = m
            print("✅ Cattle breed model loaded and ready!")
            return breed_model

        except Exception as err1:
            print(f"⚠️ Primary model load notice: {err1}. Trying standard architecture...")
            try:
                _base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
                _base.trainable = False

                m = models.Sequential([
                    _base,
                    layers.GlobalAveragePooling2D(),
                    layers.Dense(128, activation="relu"),
                    layers.Dropout(0.3),
                    layers.Dense(5, activation="softmax"),
                ])
                m.load_weights(str(BASE_DIR / "cattle_breed_model.h5"))
                m.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                breed_model = m
                print("✅ Fallback model loaded and ready!")
                return breed_model
            except Exception as err2:
                print(f"❌ Critical model load failure: {err2}")
                raise err2


# Warm up model in background so the HTTP port binds immediately on startup
threading.Thread(target=load_breed_model, daemon=True).start()

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
    """Health check endpoint used by Render / cloud monitors."""
    return jsonify({
        "status": "healthy",
        "model_loaded": breed_model is not None
    }), 200


# ── Prediction API ────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Lightning-fast cattle breed prediction (< 1 second).
    Rejects non-cattle images (humans, dogs, objects) if confidence < MIN_CONFIDENCE.
    """
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]

        # Ensure model is ready
        model = load_breed_model()

        # Read and resize image directly in memory
        img = PILImage.open(file.stream).convert("RGB").resize((224, 224))
        img_arr = np.array(img, dtype="float32") / 255.0
        x = np.expand_dims(img_arr, axis=0)

        # Predict
        preds = model.predict(x, verbose=0)[0]
        sorted_probs = np.sort(preds)
        top1_prob = sorted_probs[-1]
        top2_prob = sorted_probs[-2]
        confidence = round(float(top1_prob) * 100, 2)
        idx = int(np.argmax(preds))

        # Margin check: on clear cattle images, the top class dominates (>85% and margin > 40%)
        margin = float(top1_prob - top2_prob)

        if confidence < MIN_CONFIDENCE or margin < 0.40:
            return jsonify({
                "error": f"This does not appear to be a cattle image (confidence {confidence}% is too low). "
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
