FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install PyTorch with CUDA support first (Done BEFORE copying requirements.txt to cache the 750MB download)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchaudio==2.2.2 \
    --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .

# Install remaining packages
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm || true

COPY . .

EXPOSE 8000
