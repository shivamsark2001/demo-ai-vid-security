# RunPod Serverless Worker for Video Security Analysis
# 2-Stage Pipeline: SigLIP2 Edge Detection + Gemini Verification

FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

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

# Pre-download SigLIP2 model (ViT-SO400M-14-SigLIP-384) via open_clip
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli')"

# Copy handler
COPY handler.py .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
