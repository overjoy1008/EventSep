# Docker
---
### Dockerfile
```
Dockerfile
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

```

### Create Docker Image
docker build -t eventsep-nightly:latest .

### Create Docker Container
docker run --gpus all -it --rm \
    --name eventsep_nightly \
    -v /mnt/c/Users/EP800-202H/Gradient/Github/EventSep:/workspace \
    -w /workspace/AudioSep \
    eventsep-nightly:latest \
    /bin/bash



# AudioSep
---
### Inference
```
python
# run_inference.py

from pipeline import build_audiosep, separate_audio
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_audiosep(
    config_yaml="config/audiosep_base.yaml",
    checkpoint_path="checkpoint/audiosep_base_4M_steps.ckpt",
    device=device,
)

audio_file = "input/lullaby_short.mp3"
text = "female vocal only"
output_file = "output/lullaby_female_vocal_only.wav"

# ----------------------------------------------
# AudioSep processes the audio at 32 kHz sampling rate
separate_audio(model, audio_file, text, output_file, device)

```

### Additional Terminal Commands
`pip install lightning==2.4.0 pytorch-lightning==2.4.0 lightning-utilities==0.11.2`

### Changes
- AudioSep/models/CLAP/open_clip/factory.py
    - *In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`*