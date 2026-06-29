FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -U openmim==0.3.9 && \
    mim install "mmengine==0.10.7" && \
    mim install "mmcv==2.2.0"

RUN python -m pip install --no-cache-dir \
    scikit-image==0.26.0 \
    evaluate==0.4.6 \
    ftfy==6.3.1 \
    regex==2026.2.28 \
    loralib==0.1.2 \
    setproctitle==1.3.7
