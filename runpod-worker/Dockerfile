# RunPod Serverless Worker for Video Security Analysis
# 2-Stage Pipeline: SigLIP2 Edge Detection + Gemini Verification

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set cache directories EARLY so model downloads go to the right place
ENV HF_HOME=/app/models
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

# Pre-download and cache SigLIP2 model (saves ~60s on cold start)
# Model: ViT-SO400M-14-SigLIP-384 (~1.5GB)
RUN python -c "\
import open_clip; \
print('Downloading SigLIP2 model...'); \
model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli'); \
print('Model downloaded and cached!'); \
"

# Copy handler
COPY handler.py .

# Runtime environment
ENV PYTHONUNBUFFERED=1

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
