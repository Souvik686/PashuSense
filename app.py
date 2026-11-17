"""
PashuSense - Production Flask Server
====================================
Optimized for local and cloud deployment (Render, Oracle Cloud, etc.)

Pipeline:
  Stage 1 — ImageNet pre-screen (MobileNetV2, 1000 classes)
            Detects if image contains cattle (ox, water buffalo, bison).
            Rejects dogs, humans, furniture, non-cattle objects.

  Stage 2 — Breed classifier (Custom MobileNetV2, 5 classes)
            Classifies specific cattle breed and fetches production & health data.
"""

import os
import threading
import numpy as np
from pathlib import Path

# Reduce TensorFlow log spam & thread contention on low-core instances
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf

# Limit TF threads to 1 to prevent thread deadlocks with Gunicorn on cloud CPUs
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, models
from PIL import Image as PILImage

from breed_data import breed_info

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

# ─── Settings ─────────────────────────────────────────────────────────────────
# Direct ImageNet class indices for bovines (0-indexed):
# 345: 'ox', 346: 'water_buffalo', 347: 'bison'
CATTLE_CLASS_INDICES = {345, 346, 347}
IMAGENET_TOP_K = 25

# Minimum breed confidence to return a prediction
MIN_BREED_CONFIDENCE = 55.0

# Allowed extensions for the static-file catch-all route
ALLOWED_STATIC_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp"}

# ─── Lazy Model Loading ───────────────────────────────────────────────────────
# Models are loaded in background thread or upon first prediction request.
# This allows the server to bind the HTTP port IMMEDIATELY so cloud health
# checks (Render, Heroku, etc.) pass in 1 second instead of hanging.

_screener = None
_breed_model = None
_model_lock = threading.Lock()


def get_screener():
    global _screener
    with _model_lock:
        if _screener is None:
            print("⏳ Loading ImageNet screener...")
            _screener = MobileNetV2(weights="imagenet", include_top=True)
            print("✅ ImageNet screener ready.")
        return _screener


def get_breed_model():
    global _breed_model
    with _model_lock:
        if _breed_model is None:
            print("⏳ Loading breed classifier...")
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
            _breed_model = m
            print("✅ Breed classifier ready.")
        return _breed_model


# Warm up models in a background thread so port binds immediately
def _warmup_background():
    try:
        get_screener()
        get_breed_model()
    except Exception as e:
        print(f"⚠️ Background warmup note: {e}")

threading.Thread(target=_warmup_background, daemon=True).start()

# ─── Labels ───────────────────────────────────────────────────────────────────
with open(BASE_DIR / "labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]
print(f"✅ Labels: {labels}")

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(BASE_DIR))
CORS(app)


def is_cattle(img_array: np.ndarray) -> bool:
    """
    Returns True if the image contains cattle/bovine animals.
    Uses top-K predictions of MobileNetV2 without external network lookups.
    """
    try:
        screener_model = get_screener()
        x = preprocess_input(np.expand_dims(img_array.astype("float32"), axis=0))
        preds = screener_model.predict(x, verbose=0)
        top_indices = np.argsort(preds[0])[-IMAGENET_TOP_K:]
        return bool(set(top_indices) & CATTLE_CLASS_INDICES)
    except Exception as e:
        print(f"⚠️ Screener check warning: {e}")
        # If screener encounters an issue, fallback to classifier
        return True


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def login_page():
    """Entry point — login / sign-up page."""
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    """Main dashboard (reached after login)."""
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    """Health check endpoint for cloud platforms."""
    return jsonify({"status": "healthy"}), 200


# ── Prediction API ────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Two-stage cattle breed prediction.
    Processes image completely in-memory (no disk writes).
    """
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]

        # Read directly from request stream (faster, avoids disk I/O conflicts)
        img = PILImage.open(file.stream).convert("RGB").resize((224, 224))
        img_arr = np.array(img)

        # ── Stage 1: Is it a cow?
        if not is_cattle(img_arr):
            return jsonify({
                "error": "This does not appear to be a cattle image. Please upload a clear photo of a cow."
            }), 200

        # ── Stage 2: Breed Classification
        model = get_breed_model()
        x = np.expand_dims(img_arr.astype("float32") / 255.0, axis=0)
        preds = model.predict(x, verbose=0)
        idx = int(np.argmax(preds[0]))
        confidence = round(float(np.max(preds[0])) * 100, 2)

        if confidence < MIN_BREED_CONFIDENCE:
            return jsonify({
                "error": f"Cattle detected, but breed is unclear ({confidence}% confidence). Try a closer or clearer photo."
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
        return jsonify({"error": f"Internal prediction error: {str(e)}"}), 500


# ── Static file catch-all (always last) ──────────────────────────────────────

@app.route("/<path:filename>")
def static_files(filename):
    """Serve whitelisted static assets from the project root."""
    if Path(filename).suffix.lower() not in ALLOWED_STATIC_EXT:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(BASE_DIR), filename)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 PashuSense running on port {port}")
    print(f"   Login     → http://localhost:{port}/")
    print(f"   Dashboard → http://localhost:{port}/dashboard\n")
    app.run(host="0.0.0.0", port=port, debug=False)
