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

# Pre-download and cache models (optional - will download at runtime if this fails)
# Try transformers SigLIP2 first, then open_clip as fallback
RUN python -c "\
import sys; \
try: \
    from transformers import AutoProcessor, AutoModel; \
    print('📦 Downloading SigLIP2 via transformers...'); \
    AutoProcessor.from_pretrained('google/siglip2-so400m-patch14-384'); \
    AutoModel.from_pretrained('google/siglip2-so400m-patch14-384'); \
    print('✅ SigLIP2 cached via transformers!'); \
except Exception as e: \
    print(f'⚠️ Transformers cache failed: {e}'); \
    print('Trying open_clip fallback...'); \
    try: \
        import open_clip; \
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP-384', pretrained='webli'); \
        tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384'); \
        print('✅ SigLIP cached via open_clip!'); \
    except Exception as e2: \
        print(f'⚠️ open_clip cache also failed: {e2}'); \
        print('Models will download at runtime.'); \
" || echo "Model pre-caching skipped, will download at runtime"

# Copy handler
COPY handler.py .

# Runtime settings
# Note: Don't set offline mode - allow runtime downloads if cache missed
ENV PYTHONUNBUFFERED=1

# RunPod serverless entrypoint
CMD ["python", "-u", "handler.py"]
