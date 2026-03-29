# EventSep

EventSep is a text-queried sound event separation framework that combines pretrained source separation, sound event detection, and diffusion-based refinement models. The method follows the paper `EventSep: 사전 학습 모델 결합 기반의 선택적 음원 분리` and focuses on improving semantic alignment and temporal selectivity without additional fine-tuning.

## Links

- Demo: [https://eventsep-demo.onrender.com/](https://eventsep-demo.onrender.com/)
- Paper PDF: [paper/EventSep.pdf](paper/EventSep.pdf)

## Overview

Text-guided source separation is flexible, but it often suffers from ambiguous prompts, missing temporal information, and prompt-audio mismatch. EventSep addresses this by combining:

- PANNs-based frame-wise SED for time-selective masking
- AudioSep for text-conditioned semantic separation
- FlowSep for perceptual refinement and high-frequency restoration
- STFT-based ensemble fusion for more stable final outputs

The full pipeline is:

1. Map the text prompt to the closest AudioSet-style target class.
2. Extract frame-wise event probabilities with SED.
3. Apply a temporal mask to suppress over-separation in irrelevant regions.
4. Run AudioSep and FlowSep in parallel.
5. Fuse both outputs in the STFT domain.

## Main Findings

- EventSep consistently improves semantic alignment over AudioSep on VGGSound.
- SED soft masking is the most important contributor to performance gains.
- The method preserves AudioSep's intelligibility while benefiting from FlowSep's perceptual refinement.

### Overall Results

| Model | VGGSound CLAP | VGGSound CLAP-A | VGGSound STOI | VGGSound ESTOI | MUSIC CLAP | MUSIC CLAP-A | MUSIC STOI | MUSIC ESTOI | ESC-50 CLAP | ESC-50 CLAP-A | ESC-50 STOI | ESC-50 ESTOI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AudioSep | 0.2838 | 0.7881 | 0.6408 | 0.5790 | 0.3497 | 0.8830 | 0.6973 | 0.6325 | 0.4426 | 0.7700 | 0.6976 | 0.5998 |
| FlowSep | 0.3222 | 0.7344 | 0.4278 | 0.3429 | 0.2083 | 0.7108 | 0.4534 | 0.3615 | 0.3545 | 0.6332 | 0.4569 | 0.3290 |
| EventSep (Ours) | **0.3549** | **0.8199** | 0.6403 | 0.5786 | 0.3494 | **0.8832** | 0.6969 | 0.6321 | **0.4433** | **0.7705** | **0.6977** | **0.5999** |

On VGGSound, EventSep improves CLAPScore from `0.2838` to `0.3549`, which is about a 25% relative gain over AudioSep.

### Masking Ablation

Soft masking with threshold `0.55` gave the best VGGSound ablation result:

- SDRi: `8.901`
- SI-SDR: `8.975`

### Ensemble Ablation

The ensemble variants show similar performance, with small trade-offs across CLAPScore and CLAPScoreA:

| Method | CLAPScore | CLAPScoreA |
| --- | ---: | ---: |
| Rate-weighted | 0.3396 | 0.7979 |
| Band-split | 0.3396 | 0.8009 |
| Reverse Band-split | 0.3394 | **0.8011** |
| Progressive | **0.3398** | 0.7977 |
| Reverse Progressive | 0.3387 | 0.8009 |

## Repository Scope

This repository is intentionally prepared for code sharing and demo reproducibility. Large checkpoints, datasets, generated outputs, temporary files, and legacy experimental folders are excluded from version control.

Included:

- core EventSep inference and benchmark code
- adapted `AudioSep`, `FlowSep`, and `audioset_tagging_cnn` source trees
- evaluation scripts and lightweight metadata
- one demo input file: `input/lullaby_short.mp3`
- paper PDF

Excluded from Git:

- model checkpoints such as `.ckpt`, `.pt`, `.pth`
- dataset payloads and generated evaluation audio
- inference outputs and experiment logs
- unused or paper-external tech such as `demucs/`
- nested repository metadata such as subdirectory `.git/`

## Environment

Two setup paths are included:

- `environment.yml`
- `Dockerfile`

Note: pretrained checkpoints are not committed in this repository, so they must be provided separately for full inference.

## Inference

The unified inference entrypoint is [`inference.py`](./inference.py).

Example:

```bash
python inference.py ^
  --audio input/lullaby_short.mp3 ^
  --text "female vocal" ^
  --masking soft ^
  --threshold 0.55 ^
  --base_model ensemble_a ^
  --output_audiosep output/lullaby_short_eventsep.wav ^
  --vis_dir output/visualize
```

Supported base models:

- `audiosep`
- `flowsep`
- `both`
- `ensemble_a`
- `ensemble_b`
- `ensemble_c`
- `ensemble_d`
- `ensemble_e`

## Benchmark

The benchmark entrypoint is [`benchmark.py`](./benchmark.py). By default, benchmark result folders are ignored from Git.

## Limitations

- EventSep depends on the AudioSet label space used by PANNs-SED, so fine-grained instrument distinction can be limited.
- FlowSep can improve perceptual quality, but its generative nature may sometimes introduce small artifacts not present in the original signal.
- The paper mainly evaluates single-concept prompts rather than more complex negative or context-heavy prompts.

## Citation

```bibtex
@article{eventsep,
  title={EventSep: 사전 학습 모델 결합 기반의 선택적 음원 분리},
  author={Park, Kyungbin},
  year={2026}
}
```
