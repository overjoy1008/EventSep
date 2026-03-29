# ===== EventSep Docker =====
# (RTX 3060용 CUDA 12.1)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ---- 기본 유틸 ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg git ca-certificates curl build-essential pkg-config \
    libavcodec-dev libavformat-dev libavdevice-dev \
    libavfilter-dev libavutil-dev libswscale-dev \
    python3 python3-pip python3-venv python3-dev && \
    rm -rf /var/lib/apt/lists/*

# ---- 프로젝트 ----
WORKDIR /workspace
# COPY를 실행하면 이미지 용량이 엄청 커짐
# COPY . /workspace
COPY environment.yml /workspace/

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

# ---- torch 관련 패키지 제거 ----
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
# 🚀 Stable PyTorch 설치 (CUDA 12.1 → RTX 3060 완전 지원)
# =============================================================
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

CMD ["/bin/bash"]
