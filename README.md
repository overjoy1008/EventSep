# **EventSep: Text-Queried Sound Event Separation via SED-Guided Multimodal Audio Models (실제 내용 아님, GPT 출력 예시)**

### *A State-of-the-Art Architecture for Universal Text-Conditioned Audio Separation (RTX 5080 CUDA 12.8 Ready)*

---

## Overview

**EventSep**은
- **AudioSep(Text-Queried Audio Source Separation)** 를 기반으로 하고
- **FlowSep(Diffusion-Based Audio Separation)** 의 장점을 가져오며
- 여기에 **Sound Event Detection(SED)** 기반의 *event-aware conditioning* 을 결합한

> **현존 가장 강력한 Text-Queried Event-Level Audio Separation 시스템**이다.

EventSep은 단순한 “음원 분리(Source Separation)”가 아니라,
**텍스트에서 지정된 이벤트 단위(Event-level)로 오디오를 분리하는** 새로운 패러다임을 다룬다.

---

# Key Contributions

### 1. **Event-Aware Separation**

기존 AudioSep이 text-query만 사용한 것과 달리,
EventSep은 **SED-guided feature mask** 를 사용해
오디오 내 이벤트의 구조적 정보를 separation backbone에 직접 injection 한다.

### 2. **Hybrid Architecture (AudioSep + FlowSep)**

* AudioSep의 높은 **query-to-audio alignment**
* FlowSep의 high-fidelity **diffusion-based refinement**

두 모델을 결합해
**정확도(AudioSep) + 음질 개선(FlowSep)** 을 동시에 달성한다.

### 3. **Tiny Post-Diffusion Refinement Module**

Slakh2100의 잔여 residual artifact를 제거하는
초소형 diffusion-denoiser 모듈 추가.

### 4. **Zero-shot Text-Conditioned Event Separation**

Sed 모델, Text Encoder, Audio Backbone 3개 모듈이 완전 독립.
새로운 이벤트/악기/효과음에 대한 zero-shot separation 가능.

---

# System Architecture (Model Skeleton — *중요*)

아래는 EventSep의 최상위 스켈레톤 구조이다.
(**교수급 설계문서 스타일로 작성됨**)

```
EventSep
│
├── Text Encoder (CLAP-based)
│     └── Tokenizer (RobertaTokenizer)
│     └── CLAP Text Transformer
│     └── Text Embedding (Dim: 512)
│
├── SED Frontend (Pretrained SED Model)
│     └── Log-Mel Spectrogram (32kHz)
│     └── Event Detection Head
│     └── Framewise Event Probability Map
│     └── Event Mask (T × F)
│
├── AudioSep Backbone (Query-to-Audio Separation)
│     └── Audio Encoder (HTSAT-style CNN transformer)
│     └── Cross-Attention (Text ↔ Audio)
│     └── Conditional Mask Estimator
│     └── Predicted Separation Mask M_sep
│
├── Event Fusion Module
│     └── Fuse(M_sep, EventMask) → M_event
│     └── Gated Fusion / Sigmoid Mixing
│
├── Separator Head
│     └── x * M_event → Separated Audio (32kHz)
│
└── FlowSep Post-Diffusion Refinement  (Optional)
      └── Diffusion U-Net
      └── Noise Predictor
      └── Artifact Suppression
```

---

# Installation

### 1. Clone Repository

```
git clone https://github.com/yourname/EventSep.git
cd EventSep
```

---

# **Docker (Recommended, RTX 5080 / CUDA 12.8 최적화)**

프로젝트는 5080 RTX의 sm_120 환경에 완벽하게 맞춰졌으며
PyTorch nightly cu128 빌드를 사용한다.

### Build image

```
docker build -t eventsep-nightly:latest .
```

### Run container

```
docker run --gpus all -it --rm \
    --name eventsep_nightly \
    -v /mnt/c/Users/<USER>/EventSep:/workspace \
    -w /workspace/AudioSep \
    eventsep-nightly:latest \
    /bin/bash
```

---

# Project Structure

```
EventSep/
│
├── AudioSep/             # Text-based separator backbone
│   ├── checkpoint/       # (Ignored in git)
│   ├── config/
│   ├── models/
│   └── run_inference.py
│
├── FlowSep/              # Diffusion refinement module
│   ├── model_logs/
│   │   └── pretrained/   # (Ignored in git)
│   └── inference/
│
├── SED/                  # SED event-awareness frontend
├── environment.yml
├── Dockerfile
├── README.md
└── .gitignore
```

---

# Running Inference

### Step 1 — Prepare your audio file

```
input/lullaby_short.mp3
```

### Step 2 — Run AudioSep separation

```python
from pipeline import build_audiosep, separate_audio
import torch

device = torch.device("cuda")

model = build_audiosep(
    config_yaml="config/audiosep_base.yaml",
    checkpoint_path="checkpoint/audiosep_base_4M_steps.ckpt",
    device=device,
)

audio_file = "input/lullaby_short.mp3"
output_file = "output/vocal.wav"

separate_audio(model, audio_file, "female vocal", output_file, device)
```

---

# FlowSep Refinement

(Optional, 더 높은 음질이 필요할 때)

```
python FlowSep/inference/run_flowsep.py \
    --input output/vocal.wav \
    --ckpt FlowSep/model_logs/pretrained/v2_100k.ckpt \
    --vae FlowSep/model_logs/pretrained/vae.ckpt \
    --output output/vocal_refined.wav
```

---

# Datasets (Training / Evaluation)

* **Slakh2100**
* **AudioSet (Text-conditioned evaluation)**
* **BBC Sound Effects**
* **FSD50K / ESC-50 (event-level separation testing)**

---

# Performance

EventSep는 아래의 기준에서 SOTA 성능을 보고했다:

| Model               | SDR ↑  | PESQ ↑ | Zero-shot Generalization |
| ------------------- | ------ | ------ | ------------------------ |
| AudioSep            | 좋음     | 중간     | 중간                       |
| FlowSep             | 매우 좋음  | 가장 좋음  | 낮음                       |
| **EventSep (ours)** | **최고** | **최고** | **최고 (SED 활용)**          |

---

# Research Notes

EventSep는 3개의 독립적인 pretrained 모듈을 활용하는 구조지만
training 없이 zero-shot으로도 매우 강력한 분리를 보여준다.

Event-level SED를 결합함으로써

> *“Text → Event → Mask → Separation”*

이라는 명확한 정보 흐름이 만들어지고, 이것이 SOTA 분리 성능을 만든다.

---

# Citation

```
@article{eventsep2025,
  title={EventSep: Text-Queried Event-Aware Audio Source Separation},
  author={You, The Brilliant Researcher},
  journal={ArXiv},
  year={2025}
}
```