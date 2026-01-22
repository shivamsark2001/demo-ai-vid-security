# RunPod Serverless Worker for Video Security Analysis
# Sequential Processing: Scan until first anomaly

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Set cache directories EARLY so all downloads go to /app/models
ENV HF_HOME=/app/models
ENV HF_HUB_CACHE=/app/models
ENV TRANSFORMERS_CACHE=/app/models
ENV TORCH_HOME=/app/models
ENV XDG_CACHE_HOME=/app/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Try to pre-download SigLIP model (optional - will download at runtime if this fails)
# This can timeout on slow connections, so we make it non-blocking
RUN python -c "\
import open_clip; \
print('📦 Downloading SigLIP model...'); \
model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli'); \
print('📦 Downloading tokenizer...'); \
tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384'); \
print('✅ SigLIP model cached!'); \
" || echo "⚠️ Model pre-cache failed, will download at runtime"

# Copy handler
COPY handler.py .

# Runtime settings
# Note: Don't set offline mode - allow runtime downloads if cache missed
ENV PYTHONUNBUFFERED=1

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
