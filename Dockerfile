# ===== EventSep Docker (PyTorch Nightly + CUDA 12.8, RTX 5080 sm_120) =====
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# ---- 기본 유틸 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates curl build-essential pkg-config \
    libavcodec-dev libavformat-dev libavdevice-dev \
    libavfilter-dev libavutil-dev libswscale-dev \
    python3 python3-pip python3-venv python3-dev && \
    rm -rf /var/lib/apt/lists/*

# ---- 프로젝트 ----
WORKDIR /workspace
COPY . /workspace

# ---- python venv(EventSep) ----
RUN python3 -m venv /opt/EventSep
ENV PATH="/opt/EventSep/bin:$PATH"

# ---- pip 최신화 ----
RUN pip install --upgrade pip setuptools wheel

# ---- environment.yml의 pip 목록만 추출 ----
RUN awk '/^[[:space:]]*- pip:/{flag=1;next} \
          flag && /^[[:space:]]*-/ {sub(/^[[:space:]]*-[[:space:]]*/, ""); print}' \
          /workspace/environment.yml \
        > /workspace/requirements.txt

# ---- torch 관련 패키지 제거 (nightly 설치 전에 충돌 방지) ----
RUN sed -i '/torch/d' /workspace/requirements.txt && \
    sed -i '/pytorch-lightning/d' /workspace/requirements.txt && \
    sed -i '/lightning/d' /workspace/requirements.txt && \
    sed -i '/torchmetrics/d' /workspace/requirements.txt && \
    sed -i '/torchlibrosa/d' /workspace/requirements.txt && \
    sed -i '/torchfile/d' /workspace/requirements.txt

# ---- av 버전 패치 ----
RUN sed -i 's/av==10.0.0/av==12.0.0/' /workspace/requirements.txt

# ---- pip installs: torch 제외 ----
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# =============================================================
# 🚀 핵심 라인: CUDA 12.8 + PyTorch Nightly(sm_120 지원) 설치
# =============================================================
RUN pip install --pre torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/nightly/cu128

CMD ["/bin/bash"]
