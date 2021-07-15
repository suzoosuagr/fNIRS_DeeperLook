FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN apt update && \
    apt install -y --no-install-recommends \
    git vim libgtk2.0-dev wget && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace