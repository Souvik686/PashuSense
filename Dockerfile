# ── Hugging Face Spaces Deployment ───────────────────────────────────────────
# Uses Python 3.12-slim base (compatible with TensorFlow 2.19)
# HF Spaces requires the app to listen on port 7860

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies needed by some TF/PIL packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# HF Spaces routes external traffic to port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["python", "app.py"]
