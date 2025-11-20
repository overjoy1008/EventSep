# Docker

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

### Additional Terminal Commands
pip install lightning==2.4.0 pytorch-lightning==2.4.0 lightning-utilities==0.11.2 tensorboard==2.0.0

### Changes
- AudioSep/models/CLAP/open_clip/factory.py
    - *In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`*

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

===== RESULTS: VGGSound =====
SDR: 9.7677
SDRi: 9.1437
SISDR: 9.0434
CLAPScore: 0.2838
CLAPScoreA: 0.7881

===== RESULTS: AudioCaps =====
SDR: 8.2091
SDRi: 8.2198
SISDR: 7.1885
CLAPScore: 0.3460
CLAPScoreA: 0.7624

===== RESULTS: ESC50 =====
SDR: 10.0422
SDRi: 10.0397
SISDR: 8.8109
CLAPScore: 0.3253
CLAPScoreA: 0.7774

===== RESULTS: MUSIC =====
SDR: 9.4454
SDRi: 9.3130
SISDR: 7.8845
CLAPScore: 0.3142
CLAPScoreA: 0.8638



# HTS-AT

### Additional Terminal Commands
pip install h5py==3.6.0 museval torchcontrib==0.0.2
<!-- `pip install cog` -->

### Changes
<!-- - HTS-Audio-Transformer/model/htsat.py - def forward()
    - *NotImplementedError: Padding size 2 is not supported for 4D input tensor*
    - x의 dimension이 서로 4D/3D 등 차원이 안 맞아서 에러남
    - HTS-Audio-Transformer/model/safe_stft.py를 만든 뒤 이를 대신 참고하도록 수정
- HTS-Audio-Transformer/model/safe_stft.py 코드 제작 -->

### Inference
```
python
CUDA_VISIBLE_DEVICES=0 python htsat_infer.py \
  --checkpoint checkpoints/HTSAT_AudioSet_Saved_1.ckpt \
  --audio input/lullaby_short.mp3
```



# FlowSep

### Additional Terminal Commands
pip install wandb einops timm laion-clap
<!-- `git clone https://github.com/haoheliu/AudioLDM.git`
`pip install -e AudioLDM/taming-transformers`
root@ea943bf5fb2b:/workspace/AudioLDM-training-finetuning# export PYTHONPATH=/workspace/AudioLDM-training-finetuning/taming-transformers:$PYTHONPATH -->
<!-- `pip install git+https://github.com/google-research/google-research.git#subdirectory=frechet_audio_distance` -->

### Changes
- FlowSep/lass_inference.py
    - *TypeError: _StoreFalseAction.__init__() got an unexpected keyword argument 'type'*
- FlowSep\src\utilities\audio\stft.py
    - *TypeError: pad_center() takes 1 positional argument but 2 were given*
    - *TypeError: mel() takes 0 positional arguments but 5 were given*
- https://github.com/CompVis/taming-transformers의 taming 폴더를 FlowSep/src/taming으로 위치시키기
- FlowSep\src\utilities\data\dataset.py의 read_wav_file()
    - RTX5080 Nightly 버전을 지원하는 torchcodec이 존재하지 않음 -> torchaudio.load 대신 librosa.load를 써야만 하는 상황
- FlowSep\src\latent_diffusion\models\ddpm_flow.py의 generate_sample()
    - if self.extra_channels: 부분 clause 전체 수정


===== VGGSound RESULTS (630k-best.pt) =====
SDR: -5.7053
SDRi: -6.9945
SISDR: -36.9804
CLAPScore: 0.1532
CLAPScoreA: 0.6334

===== AudioSet RESULTS (630k-best.pt) =====
NaN
NaN
NaN
NaN
NaN

===== AudioCaps RESULTS (630k-best.pt) =====

===== ESC50 RESULTS (630k-best.pt) =====
SDR: -6.9323
SDRi: -6.9348
SISDR: -40.5134
CLAPScore: 0.2118
CLAPScoreA: 0.4060

===== Music RESULTS (630k-best.pt) =====
SDR: -3.7392
SDRi: -4.1244
SISDR: -35.6030
CLAPScore: 0.2539
CLAPScoreA: 0.6504



# PANNs

### Additional Terminal Commands
pip install

### Changes

### Inference
python pytorch/inference.py sound_event_detection \
    --model_type=Cnn14_DecisionLevelMax \
    --checkpoint_path="pytorch/Cnn14_DecisionLevelMax_mAP=0.385.pth" \
    --audio_path="resources/furina.mp3" \
    --cuda

python pytorch/inference.py sound_event_detection \
    --model_type=Cnn14_DecisionLevelAtt \
    --checkpoint_path=pytorch/Cnn14_DecisionLevelAtt_mAP=0.425.pth \
    --audio_path="resources/furina.mp3" \
    --cuda

<!-- python pytorch/inference.py sound_event_detection \
    --model_type=Wavegram_Logmel_Cnn14 \
    --checkpoint_path=pytorch/Wavegram_Logmel_Cnn14_mAP=0.439.pth \
    --audio_path="resources/furina.mp3" -->

python pytorch/framewise_inference.py sound_event_detection \
    --model_type=Cnn14_DecisionLevelAtt \
    --checkpoint_path=pytorch/Cnn14_DecisionLevelAtt_mAP=0.425.pth \
    --audio_path="resources/furina.mp3" \
    --cuda

python pytorch/framewise_inference.py sound_event_detection \
    --model_type=Cnn14_DecisionLevelAtt \
    --checkpoint_path=pytorch/Cnn14_DecisionLevelAtt_mAP=0.425.pth \
    --audio_path="resources/lullaby_short.mp3" \
    --cuda



# EventSep

### Inference
python inference.py \
  --audio input/lullaby_short.mp3 \
  --text "female vocal only" \
  --output output/lullaby_short_female_vocal_only.wav

python inference.py \
  --audio input/lullaby_short.mp3 \
  --text "Lullaby" \
  --output output/lullaby_short_Lullaby.wav

python inference.py \
  --audio input/lullaby_short.mp3 \
  --text "Lullaby" \
  --output output/lullaby_short_Lullaby.wav \
  --debug

python inference.py \
    --audio input/lullaby_short.mp3 \
    --text "Lullaby" \
    --threshold 0.5 \
    --output output/lullaby_sep.wav \
    --debug

python inference.py \
    --audio input/lullaby_short.mp3 \
    --text "Lullaby" \
    --threshold 0.65 \
    --output output/lullaby_sep.wav \
    --debug


VGGSound Avg SDRi: 3.682, SISDR: 8.695, CLAPScore: 0.393, CLAPScoreA: 0.818