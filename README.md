# 🐄 PashuSense

> AI-powered cattle breed recognition prototype web app — identifies breed, milk production & health tips from a photo.

---

## How It Works

```
Upload image
     │
     ▼
Stage 1 — ImageNet MobileNetV2 (1000-class, pre-trained)
"Is there actually a cow in this image?"
     │
     ├─ No cattle detected → ❌ "This does not appear to be a cattle image"
     │   (rejects dogs, humans, objects, random photos)
     │
     └─ Cattle found
               │
               ▼
         Stage 2 — Custom MobileNetV2 (trained on 5 breeds)
         "Which breed is it?"
               │
               ├─ Confidence < 55% → ❌ "Breed unclear, try a clearer photo"
               └─ ✅ Breed + Confidence + Milk Production + Health Tips
```

---

## Quick Start (Google Colab)

1. Open `Cattles__Recognization.ipynb` in [Google Colab](https://colab.research.google.com)
2. Upload these **4 files** when the first cell runs:

   | File | Purpose |
   |---|---|
   | `cattle_breed_model.h5` | Trained model weights |
   | `breed_data.py` | Breed metadata |
   | `labels.txt` | 5 breed class names |
   | `index.html` | Web UI template |

3. Run all cells top to bottom
4. Copy the **ngrok URL** from the last cell output and open it in your browser

> ⚠️ Replace `YOUR_NGROK_AUTH_TOKEN` in the last cell with your own token from [ngrok.com](https://ngrok.com/signup) (free).

---

## Quick Start (Local)

> Requires **Python 3.9–3.12** — TensorFlow does not support Python 3.13 yet.

```bash
pip install -r requirements.txt
python app.py
# Open: http://localhost:8000
```

---

## Breeds Supported

| Breed | Milk Production | Health Tip |
|---|---|---|
| Ayrshire cattle | 6,000 L/year | Well-ventilated barn, balanced feed |
| Brown Swiss cattle | 7,500 L/year | Regular exercise, check for mastitis |
| Holstein Friesian cattle | 8,000 L/year | High-quality roughage, hygiene |
| Jersey cattle | 5,500 L/year | Monitor calcium intake |
| Red Dane cattle | 6,500 L/year | Clean water, vaccination |

---

## Project Files

| File | Purpose |
|---|---|
| `app.py` | Local Flask server (dual-stage pipeline) |
| `requirements.txt` | Python dependencies |
| `breed_data.py` | Breed metadata dictionary |
| `labels.txt` | 5 breed class names |
| `cattle_breed_model.h5` | Trained MobileNetV2 weights *(included, 11 MB)* |
| `index.html` | Main dashboard UI |
| `login.html` | Login / sign-up page |
| `plogo.png` | App logo |
| `Cattles__Recognization.ipynb` | Google Colab notebook |

> `Cattle Breeds/` training dataset and `Cattles/` sample folder are excluded from the repo (see `.gitignore`).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Cattle screening | MobileNetV2 (ImageNet pretrained, 1000 classes) |
| Breed classification | MobileNetV2 (custom trained, 5 classes) |
| Backend | Flask + Flask-CORS |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Colab tunneling | ngrok / pyngrok |
