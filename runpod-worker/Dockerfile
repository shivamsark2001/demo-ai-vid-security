# RunPod Serverless Worker for Video Security Analysis
# Python + ffmpeg + SigLIP + Gemini verification

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SigLIP model to bake into image
RUN python -c "from transformers import AutoProcessor, AutoModel; AutoProcessor.from_pretrained('google/siglip-base-patch16-384'); AutoModel.from_pretrained('google/siglip-base-patch16-384')"

# Copy handler
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
