FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV MODEL_CACHE_SIZE=5
ENV ACTIVE_MODELS=Qwen/Qwen3-TTS-12Hz-1.7B-Base
ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    git ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only dependency files first so the pip install layer is cached
# independently of source code changes. As long as these files don't
# change, Docker reuses the cached layer and skips the heavy download.
COPY demo/requirements.txt demo/requirements.txt
COPY pyproject.toml pyproject.toml

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchaudio \
    && python3 -m pip install "faster-qwen3-tts[demo]" \
    && python3 -m pip install -r demo/requirements.txt \
    && python3 -m pip install --no-deps "qwen-tts>=0.1.1"

# Copy source code last — changes to server.py, index.html, etc.
# only invalidate this thin layer, not the pip install above.
COPY . /app

EXPOSE 7860
CMD ["python3", "demo/server.py", "--host", "0.0.0.0"]
