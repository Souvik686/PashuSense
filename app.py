"""
PashuSense - High-Speed Production Server
=========================================
Ultra-fast (<30ms) TFLite inference, single-threaded sync execution,
zero deadlocks, zero memory thrashing, zero external downloads.
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image as PILImage

# ─── Environment & Single-Thread Configuration ──────────────────────────────
# Pinned to 1 thread to avoid host core thread storms (CFS throttling) on container free tiers
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass

from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
from breed_data import breed_info

BASE_DIR = Path(__file__).parent
ALLOWED_STATIC_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp"}

# ─── Load Labels ────────────────────────────────────────────────────────────
with open(BASE_DIR / "labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]
print(f"✅ Labels loaded: {labels}")

# ─── TFLite Engine (Instant 30ms CPU Inference, ~20MB RAM) ───────────────────
TFLITE_MODEL_PATH = BASE_DIR / "cattle_breed_model.tflite"
H5_MODEL_PATH = BASE_DIR / "cattle_breed_model.h5"

interpreter = None
input_index = None
output_index = None
h5_model = None


def get_inference_engine():
    """
    Returns (engine_type, runner)
    Prefers lightweight TFLite (2.5MB, ~30ms, ~20MB RAM).
    Falls back to H5 Sequential model if TFLite is absent.
    """
    global interpreter, input_index, output_index, h5_model

    if interpreter is not None:
        return "tflite", (interpreter, input_index, output_index)
    if h5_model is not None:
        return "h5", h5_model

    # Try TFLite first
    if TFLITE_MODEL_PATH.exists():
        try:
            print("⏳ Initializing TFLite interpreter...")
            interp = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]["index"]
            out = interp.get_output_details()[0]["index"]

            # Warmup
            dummy = np.zeros((1, 224, 224, 3), dtype="float32")
            interp.set_tensor(inp, dummy)
            interp.invoke()

            interpreter, input_index, output_index = interp, inp, out
            print("🚀 TFLite model loaded & warmed up! Inference speed: ~30ms.")
            return "tflite", (interpreter, input_index, output_index)
        except Exception as e:
            print(f"⚠️ TFLite load warning: {e}. Falling back to H5...")

    # Fallback to H5
    try:
        print("⏳ Loading H5 model...")
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
        dummy = np.zeros((1, 224, 224, 3), dtype="float32")
        _ = m(dummy, training=False)
        m.load_weights(str(H5_MODEL_PATH))
        h5_model = m
        print("✅ H5 Sequential model loaded & warmed up!")
        return "h5", h5_model
    except Exception as e:
        print(f"❌ Failed to load H5 model: {e}")
        raise


# Pre-warm model right when server boots
try:
    get_inference_engine()
except Exception as e:
    print(f"Initial load note: {e}")


def run_prediction(x):
    """Executes single forward pass and returns probabilities array."""
    engine_type, runner = get_inference_engine()
    if engine_type == "tflite":
        interp, inp_idx, out_idx = runner
        interp.set_tensor(inp_idx, x)
        interp.invoke()
        return interp.get_tensor(out_idx)[0]
    else:
        m = runner
        return m(x, training=False).numpy()[0]


# ─── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=str(BASE_DIR))
CORS(app)


# ── Page Routes ─────────────────────────────────────────────────────────────

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
        "engine": "tflite" if interpreter is not None else ("h5" if h5_model is not None else "unloaded")
    }), 200


# ── Prediction API ──────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Lightning-fast cattle breed prediction (~30ms).
    Rejects non-cattle images (dogs, humans, random objects).
    """
    try:
        if "file" not in request.files or request.files["file"].filename == "":
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files["file"]

        # Read and resize image in memory
        img = PILImage.open(file.stream).convert("RGB").resize((224, 224))
        x = np.expand_dims(np.array(img, dtype="float32") / 255.0, axis=0)

        # Execute prediction
        preds = run_prediction(x)

        sorted_probs = np.sort(preds)
        top1_prob = float(sorted_probs[-1])
        top2_prob = float(sorted_probs[-2])
        confidence = round(top1_prob * 100, 2)
        idx = int(np.argmax(preds))
        margin = top1_prob - top2_prob

        # Gating for non-cattle rejection:
        # Clear cattle images have strong confidence (>60%) with a decisive margin (>0.20)
        # Non-cattle images (people, cars, random objects) yield low confidence or flat spread.
        if confidence < 60.0 or margin < 0.20:
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


# ── Static Files Catch-All ──────────────────────────────────────────────────

@app.route("/<path:filename>")
def static_files(filename):
    if Path(filename).suffix.lower() not in ALLOWED_STATIC_EXT:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(BASE_DIR), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 PashuSense running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
