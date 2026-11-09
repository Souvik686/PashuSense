"""
PashuSense - Local Flask Server
================================
Run with:  python app.py
Open:      http://localhost:8000

Two-stage prediction pipeline (no extra packages beyond requirements.txt):

  Stage 1 — ImageNet pre-screen (MobileNetV2, 1000 classes)
            Checks if the image actually contains a cow/cattle.
            Rejects dogs, humans, objects, random photos.

  Stage 2 — Breed classifier (MobileNetV2, 5 classes, your trained weights)
            Identifies the specific cattle breed.

Routes:
  GET  /             → login.html
  GET  /dashboard    → index.html
  POST /predict      → prediction API (returns JSON)
  GET  /<filename>   → static assets (plogo.png, etc.)
"""

import numpy as np
from pathlib import Path

from flask import Flask, request, render_template, send_from_directory, jsonify
from flask_cors import CORS
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions,
)
from tensorflow.keras import layers, models
from PIL import Image as PILImage

from breed_data import breed_info

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
UPLOAD_PATH = BASE_DIR / "uploaded.jpg"

# ─── Settings ─────────────────────────────────────────────────────────────────
# ImageNet synsets that count as cattle (Keras ImageNet class IDs)
CATTLE_SYNSETS = {
    "n02403003",   # ox / domestic cattle (dairy cows, beef cattle, all breeds)
    "n02408429",   # water buffalo
}

# How many top ImageNet predictions to check for cattle
IMAGENET_TOP_K = 10

# Minimum breed confidence to return a result (rejects unclear cattle photos)
MIN_BREED_CONFIDENCE = 55.0

# Allowed extensions for the static-file catch-all route
ALLOWED_STATIC_EXT = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp"}

# ─── Stage 1: ImageNet screener ───────────────────────────────────────────────
# Full MobileNetV2 (1000-class ImageNet) — knows what a real cow looks like.
# Rejects images where no cattle-related class appears in the top-K predictions.
print("⏳ Loading ImageNet screener...")
screener = MobileNetV2(weights="imagenet", include_top=True)
print("✅ ImageNet screener ready.")

# ─── Stage 2: Cattle breed classifier ────────────────────────────────────────
# Your custom MobileNetV2 (trained on 5 cattle breeds).
# The .h5 file has WEIGHTS ONLY — architecture must be built first.
print("⏳ Loading breed classifier...")
_base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
_base.trainable = False

breed_model = models.Sequential([
    _base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(5, activation="softmax"),
])
breed_model.load_weights(str(BASE_DIR / "cattle_breed_model.h5"))
breed_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
print("✅ Breed classifier ready.")

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
    Uses full ImageNet MobileNetV2 — checks top-K predictions for cattle synsets.
    img_array: uint8 numpy array shape (224, 224, 3)
    """
    # MobileNetV2 expects [-1, 1] range via preprocess_input
    x = preprocess_input(np.expand_dims(img_array.astype("float32"), axis=0))
    preds = screener.predict(x, verbose=0)
    top_k = decode_predictions(preds, top=IMAGENET_TOP_K)[0]
    # top_k is a list of (synset_id, label, probability)
    return any(synset_id in CATTLE_SYNSETS for synset_id, _, _ in top_k)


# ── Page routes ───────────────────────────────────────────────────────────────

@app.route("/")
def login_page():
    """Entry point — login / sign-up page."""
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    """Main dashboard (reached after login)."""
    return render_template("index.html")


# ── Prediction API ────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Two-stage cattle breed prediction.

    POST /predict  (multipart/form-data, field: 'file')

    Stage 1 — ImageNet screen:  is there actually a cow in this image?
    Stage 2 — Breed classify:   which of the 5 breeds is it?

    Success: { breed, confidence, milk_production, health_tips }
    Failure: { error: "..." }
    """
    # Validate upload
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No file uploaded."}), 400

    # Save uploaded file
    request.files["file"].save(str(UPLOAD_PATH))

    # Load & resize to 224×224
    img      = PILImage.open(str(UPLOAD_PATH)).convert("RGB").resize((224, 224))
    img_arr  = np.array(img)   # shape (224, 224, 3), dtype uint8

    # ── Stage 1: Does this image contain a cow? ───────────────────────────────
    if not is_cattle(img_arr):
        return jsonify({
            "error": "This does not appear to be a cattle image. "
                     "Please upload a clear, close-up photo of a cow."
        })

    # ── Stage 2: Which breed? ─────────────────────────────────────────────────
    # Breed model uses [0, 1] normalization (different from screener!)
    x          = np.expand_dims(img_arr / 255.0, axis=0)
    preds      = breed_model.predict(x, verbose=0)
    idx        = int(np.argmax(preds[0]))
    confidence = round(float(np.max(preds[0])) * 100, 2)

    if confidence < MIN_BREED_CONFIDENCE:
        return jsonify({
            "error": f"Cattle detected but breed is unclear ({confidence}% confidence). "
                     "Try a closer or clearer photo of the cow."
        })

    breed = labels[idx]
    info  = breed_info.get(breed, {"milk_production": "N/A", "health_tips": "N/A"})

    return jsonify({
        "breed":           breed,
        "confidence":      confidence,
        "milk_production": info["milk_production"],
        "health_tips":     info["health_tips"],
    })


# ── Static file catch-all (always last) ──────────────────────────────────────

@app.route("/<path:filename>")
def static_files(filename):
    """Serve whitelisted static assets from the project root."""
    if Path(filename).suffix.lower() not in ALLOWED_STATIC_EXT:
        return jsonify({"error": "Not found"}), 404
    return send_from_directory(str(BASE_DIR), filename)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🚀 PashuSense running at http://localhost:8000")
    print("   Login     → http://localhost:8000/")
    print("   Dashboard → http://localhost:8000/dashboard\n")
    app.run(host="0.0.0.0", port=8000, debug=False)
