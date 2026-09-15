FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    nano \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Keep the installer compatible with the archived AI Economist dependencies.
RUN python3.8 -m pip install --no-cache-dir --upgrade "pip<25"

# Install AI Economist and standard ML stack with a high timeout
COPY requirements.txt /workspace/requirements.txt
RUN python3.8 -m pip install --default-timeout=10000 --no-cache-dir -r requirements.txt

# Provision ~/.env virtual environment
RUN python3.8 -m venv --without-pip --system-site-packages /root/.env

COPY src /workspace/src
COPY configs /workspace/configs
COPY test.py /workspace/test.py
ENV PYTHONPATH=/workspace/src
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["bash"]
