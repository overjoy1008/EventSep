# ====================== evaluate_audiocaps_eventsep.py ======================

import os
import sys
import csv
import random
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm
import lightning.pytorch as pl  # noqa: F401

# ====== AudioSep imports ======
sys.path.append(os.path.join(os.getcwd(), "AudioSep"))
from models.clap_encoder import CLAP_Encoder  # noqa: E402

# ====== PANNs imports ======
sys.path.append("./audioset_tagging_cnn/pytorch")
from audioset_tagging_cnn.pytorch.models import (  # noqa: E402
    Cnn14_DecisionLevelAtt,
)

# ====== EventSep / common utils ======
from utils.sdr_utils import calculate_sdr, calculate_sisdr
from utils.text_utils import load_audioset_labels, select_target_class
from utils.sed_utils import extract_mask_from_panns
from utils.ensemble_utils import blend_ensemble
from utils.clapscore_utils import load_clapscore_model, load_auto_clap_model

# ====== 추가: 확장 평가 지표 (STOI / PESQ / LMSD 등) ======
from utils.metric_utils import (
    compute_stoi,
    compute_estoi,
    compute_pesq,
    compute_lmsd,
    compute_vggish_distance,    # 추가
    compute_fad,                # 추가
    compute_mrstft,             # 추가
)


class AudioCapsEvaluator:
    """
    EventSep용 AudioCaps 평가기.

    - PANNs(Cnn14_DecisionLevelAtt)를 이용한 SED-guided masking
    - AudioSep + FlowSep 듀얼 분리 후 ensemble
    - 선택적으로 Demucs 전단 분리(use_demucs)
    - CLAPScore backbone 선택 가능 (VGGSoundEventSepEvaluator와 동일)
    """

    def __init__(
        self,
        query: str = "caption",           # "caption" or "labels"
        sampling_rate: int = 32000,
        sed_threshold: float = 0.4,
        mask_mode: str = "soft",          # "soft" / "hard" / "none"
        ensemble: str = "audiosep",       # "audiosep" / "flowsep" / "ensemble_a" / ...
        ensemble_rate: float = 0.3,
        ensemble_freq: int = 4000,
        clapscore_type: str = "audiosep", # VGGSound evaluator와 동일 옵션
        audiocaps_csv_path: str = "evaluation/metadata/audiocaps_eval.csv",
        audiocaps_audio_dir: str = "evaluation/data/audiocaps",
        debug_num_samples: int = 5,
        debug_dir: str = "evaluation/debug/audiocaps_eventsep",
        embedding_model_type: str = "minilm",   # AMTH
        demo_mode: bool = False,
        use_demucs: bool = False,
        demucs_root: str = "demucs",
        demucs_model: str = "htdemucs",
    ) -> None:
        self.query = query
        self.sr = sampling_rate
        self.sed_threshold = sed_threshold
        self.mask_mode = mask_mode
        self.ensemble = ensemble
        self.ensemble_rate = ensemble_rate
        self.ensemble_freq = ensemble_freq
        self.clapscore_type = clapscore_type
        self.embedding_model_type = embedding_model_type
        self.demo_mode = demo_mode

        # ----------------- Demucs 관련 설정 -----------------
        self.use_demucs = use_demucs
        self.demucs_root = demucs_root
        self.demucs_model = demucs_model

        # ----------------- AudioCaps 메타 -----------------
        with open(audiocaps_csv_path, "r") as f:
            rows = [row for row in csv.reader(f)][1:]  # header 제거

        if self.demo_mode:
            self.eval_list = rows[:10]
        else:
            self.eval_list = rows

        self.audio_dir = audiocaps_audio_dir

        # ----------------- Device -----------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # =================================================
        #              Separation용 AudioSep CLAP
        # =================================================
        self.clap_sep = CLAP_Encoder().eval().to(self.device)

        # =================================================
        #                        PANNs
        # =================================================
        self.panns = Cnn14_DecisionLevelAtt(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        ckpt = "./audioset_tagging_cnn/pytorch/Cnn14_DecisionLevelAtt_mAP=0.425.pth"
        state = torch.load(ckpt, map_location=self.device)
        self.panns.load_state_dict(state["model"])
        self.panns = self.panns.to(self.device).eval()

        # AudioSet label 목록 (PANNs framewise에서 index 매핑용)
        self.labels = load_audioset_labels()

        # =================================================
        #                  CLAPScore 전용 모델
        # =================================================
        if self.clapscore_type == "auto":
            self.clap_score_text, self.clap_score_audio = load_auto_clap_model(
                device=self.device
            )
            self.clap_score = None
        else:
            self.clap_score = load_clapscore_model(
                model_type=self.clapscore_type, device=self.device
            )
            self.clap_score_text = None
            self.clap_score_audio = None

        # =================================================
        #                        Debug
        # =================================================
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)

        self.debug_samples = set()
        if len(self.eval_list) > debug_num_samples > 0:
            selected = random.sample(self.eval_list, debug_num_samples)
            # 첫 컬럼이 idx
            self.debug_samples = set([row[0] for row in selected])

        print(f"[AudioCaps-EventSep] Using CLAPScore model: {self.clapscore_type}")
        print(f"[AudioCaps-EventSep] Eval pairs: {len(self.eval_list)}")
        print(f"[AudioCaps-EventSep] Debug samples: {len(self.debug_samples)} "
              f"will be saved in '{self.debug_dir}'")
        if self.use_demucs:
            print(f"[AudioCaps-EventSep] Demucs semantic routing ENABLED "
                  f"(root={self.demucs_root}, model={self.demucs_model})")

    # =====================================================================
    #                        Demucs helper
    # =====================================================================

    def _run_demucs_stem(self, mix_path: str, target: str) -> np.ndarray:
        """
        Demucs 안전 실행:
        - 원본 mix_path를 tmp 파일로 복사
        - Demucs 실행 후 tmp_output 폴더에서 stem 추출
        """
        import uuid
        import shutil

        tmp_id = str(uuid.uuid4())[:8]
        tmp_in = f"/tmp/audiocaps_demucs_{tmp_id}.wav"
        tmp_out = f"/tmp/audiocaps_demucs_out_{tmp_id}"

        os.makedirs(tmp_out, exist_ok=True)
        shutil.copyfile(mix_path, tmp_in)

        cmd = [
            "python3", "-m", "demucs.separate",
            "-n", self.demucs_model,
            "--device", "cuda" if self.device == "cuda" else "cpu",
            "-o", tmp_out,
            tmp_in,
        ]
        subprocess.run(cmd, check=True, cwd=self.demucs_root)

        base = Path(tmp_in).stem
        stem_dir = os.path.join(tmp_out, self.demucs_model, base)
        stem_path = os.path.join(stem_dir, f"{target}.wav")

        if not os.path.exists(stem_path):
            raise FileNotFoundError(f"[Demucs] Stem file not found: {stem_path}")

        wav, sr = librosa.load(stem_path, sr=self.sr, mono=True)
        return wav.astype(np.float32)

    # =====================================================================
    #                           Evaluation Loop
    # =====================================================================

    def __call__(self, pl_audiosep: pl.LightningModule, pl_flowsep: pl.LightningModule) -> Dict:
        """
        pl_audiosep : AudioSep LightningModule
        pl_flowsep  : FlowSep LightningModule (ss_model에서 'waveform', 'random_start' 반환)
        """

        sdr_list, sdri_list, sisdr_list = [], [], []
        clap_list, clapA_list = [], []

        stoi_list, estoi_list = [], []
        pesq_list, lmsd_list = [], []
        vggish_list, fad_list, mrstft_list = [], [], []

        pl_audiosep.eval()
        pl_flowsep.eval()

        with torch.no_grad():
            for idx, caption, labels, _, _ in tqdm(self.eval_list):
                source_path = os.path.join(self.audio_dir, f"segment-{idx}.wav")
                mixture_path = os.path.join(self.audio_dir, f"mixture-{idx}.wav")

                if not (os.path.exists(source_path) and os.path.exists(mixture_path)):
                    continue

                source, _ = librosa.load(source_path, sr=self.sr, mono=True)
                mixture, _ = librosa.load(mixture_path, sr=self.sr, mono=True)

                # 길이 맞추기
                L0 = min(len(source), len(mixture))
                source = source[:L0]
                mixture = mixture[:L0]

                sdr_no_sep = calculate_sdr(source, mixture)

                # ---------- Text query ----------
                if self.query == "caption":
                    text_query = caption
                elif self.query == "labels":
                    text_query = labels
                else:
                    raise ValueError(f"Unknown query type: {self.query}")

                # ----------------- Demucs semantic routing -----------------
                demucs_sep = None

                cid = select_target_class(
                    labels=self.labels,
                    text_prompt=text_query,
                    embedding_model_type=self.embedding_model_type,
                    clap_encoder=self.clap_sep,
                )

                mixture_for_eventsep = mixture

                if (
                    self.use_demucs
                    and isinstance(cid, dict)
                    and "demucs_target" in cid
                ):
                    demucs_target = cid["demucs_target"]
                    try:
                        demucs_sep = self._run_demucs_stem(mixture_path, demucs_target)
                        Ld = min(len(demucs_sep), len(mixture), len(source))
                        demucs_sep = demucs_sep[:Ld]
                        mixture = mixture[:Ld]
                        source = source[:Ld]

                        mixture_for_eventsep = mixture - demucs_sep
                        print(f"[Demucs] Routed prompt '{text_query}' → stem '{demucs_target}'")
                    except Exception as e:
                        print(f"[Demucs] Failed to run Demucs for '{text_query}': {e}")
                        demucs_sep = None
                        mixture_for_eventsep = mixture

                # ----------------- SED-guided masking -----------------
                if self.mask_mode != "none":
                    cid_sed = cid
                    if isinstance(cid_sed, dict):
                        cid_sed = None

                    if cid_sed is None:
                        mixture_sed = mixture_for_eventsep
                    else:
                        mask = extract_mask_from_panns(
                            self.panns,
                            mixture_for_eventsep,
                            cid_sed,
                            threshold=self.sed_threshold,
                            device=self.device,
                        )
                        mixture_sed = mixture_for_eventsep * mask
                else:
                    mixture_sed = mixture_for_eventsep

                mix_tensor = torch.tensor(mixture_sed, dtype=torch.float32)[None, None].to(self.device)

                # =====================================================
                #                     AudioSep 추론
                # =====================================================
                cond = pl_audiosep.query_encoder.get_query_embed(
                    modality="text",
                    text=[text_query],
                    device=self.device,
                )
                sep_a = pl_audiosep.ss_model({
                    "mixture": mix_tensor,
                    "condition": cond,
                })["waveform"]
                sep_a = sep_a.squeeze().detach().cpu().numpy()

                # =====================================================
                #                     FlowSep 추론
                # =====================================================
                out_f = pl_flowsep.ss_model({"mixture": mix_tensor, "text": text_query})
                sep_f16 = out_f["waveform"].squeeze().detach().cpu().numpy()
                random_start16 = int(out_f["random_start"])

                sep_f = librosa.resample(sep_f16, orig_sr=16000, target_sr=self.sr)
                Lf = len(sep_f)

                start32 = int(random_start16 * (self.sr / 16000.0))

                if len(sep_a) < start32 + Lf:
                    pad_len = start32 + Lf - len(sep_a)
                    sep_a = np.concatenate([sep_a, np.zeros(pad_len, dtype=sep_a.dtype)])

                seg_a = sep_a[start32:start32 + Lf]

                # =====================================================
                #                         Ensemble
                # =====================================================
                seg_e = blend_ensemble(
                    seg_a,
                    sep_f,
                    mode=self.ensemble,
                    rate=self.ensemble_rate,
                    cutoff=self.ensemble_freq,
                    sr=self.sr,
                )

                sep = np.copy(sep_a)
                sep[start32:start32 + len(seg_e)] = seg_e

                # ----------------- Demucs residual fusion -----------------
                if demucs_sep is not None:
                    L = min(len(source), len(sep), len(mixture), len(demucs_sep))
                    source = source[:L]
                    mixture = mixture[:L]
                    sep = sep[:L]
                    demucs_sep = demucs_sep[:L]
                    sep = sep + demucs_sep
                else:
                    L = min(len(source), len(sep), len(mixture))
                    source = source[:L]
                    sep = sep[:L]
                    mixture = mixture[:L]

                # =====================================================
                #                         Metrics
                # =====================================================
                sdr_sep = calculate_sdr(source, sep)
                sdri_list.append(sdr_sep - sdr_no_sep)
                sdr_list.append(sdr_sep)

                sisdr_sep = calculate_sisdr(source, sep)
                sisdr_list.append(sisdr_sep)

                stoi_val = compute_stoi(source, sep, self.sr)
                estoi_val = compute_estoi(source, sep, self.sr)
                pesq_val = compute_pesq(source, sep, self.sr)
                lmsd_val = compute_lmsd(source, sep, self.sr)

                stoi_list.append(stoi_val)
                estoi_list.append(estoi_val)
                pesq_list.append(pesq_val)
                lmsd_list.append(lmsd_val)

                vggish_val = compute_vggish_distance(source, sep, self.sr)
                fad_val = compute_fad(source, sep, self.sr)
                mrstft_val = compute_mrstft(source, sep, self.sr)

                vggish_list.append(vggish_val)
                fad_list.append(fad_val)
                mrstft_list.append(mrstft_val)

                # ----------------- CLAPScore -----------------
                sep_t = torch.tensor(sep, dtype=torch.float32, device=self.device)[None]
                src_t = torch.tensor(source, dtype=torch.float32, device=self.device)[None]

                if self.clapscore_type == "auto":
                    sep_emb_text = self.clap_score_text.get_audio_embedding_from_data(
                        sep_t, use_tensor=True
                    )
                    text_emb = self.clap_score_text.get_text_embedding(
                        [text_query], use_tensor=True
                    )
                    clap_val = torch.nn.functional.cosine_similarity(
                        sep_emb_text, text_emb
                    ).item()

                    sep_emb_aud = self.clap_score_audio.get_audio_embedding_from_data(
                        sep_t, use_tensor=True
                    )
                    src_emb_aud = self.clap_score_audio.get_audio_embedding_from_data(
                        src_t, use_tensor=True
                    )
                    clapA_val = torch.nn.functional.cosine_similarity(
                        sep_emb_aud, src_emb_aud
                    ).item()
                else:
                    if isinstance(self.clap_score, CLAP_Encoder):
                        sep_emb = self.clap_score._get_audio_embed(sep_t)
                        src_emb = self.clap_score._get_audio_embed(src_t)
                        text_emb = self.clap_score._get_text_embed([text_query])
                    else:
                        sep_emb = self.clap_score.get_audio_embedding_from_data(
                            sep_t, use_tensor=True
                        )
                        src_emb = self.clap_score.get_audio_embedding_from_data(
                            src_t, use_tensor=True
                        )
                        text_emb = self.clap_score.get_text_embedding(
                            [text_query], use_tensor=True
                        )

                    clap_val = torch.nn.functional.cosine_similarity(
                        sep_emb, text_emb
                    ).item()
                    clapA_val = torch.nn.functional.cosine_similarity(
                        sep_emb, src_emb
                    ).item()

                clap_list.append(clap_val)
                clapA_list.append(clapA_val)

                # ----------------- Debug Save -----------------
                if idx in self.debug_samples:
                    base = os.path.join(self.debug_dir, f"{idx}")
                    os.makedirs(os.path.dirname(base), exist_ok=True)

                    sf.write(base + "_mix.wav", mixture, self.sr)
                    sf.write(base + "_src.wav", source, self.sr)
                    sf.write(base + "_sep.wav", sep, self.sr)
                    sf.write(base + "_audiosep.wav", sep_a[:L], self.sr)
                    sf.write(base + "_flowsep.wav", sep_f[: min(Lf, L)], self.sr)
                    if demucs_sep is not None:
                        sf.write(base + "_demucs.wav", demucs_sep[:L], self.sr)

        return dict(
            SDR=float(np.mean(sdr_list)),
            SDRi=float(np.mean(sdri_list)),
            SISDR=float(np.mean(sisdr_list)),
            CLAPScore=float(np.mean(clap_list)),
            CLAPScoreA=float(np.mean(clapA_list)),
            STOI=float(np.nanmean(stoi_list)) if len(stoi_list) > 0 else np.nan,
            ESTOI=float(np.nanmean(estoi_list)) if len(estoi_list) > 0 else np.nan,
            PESQ=float(np.nanmean(pesq_list)) if len(pesq_list) > 0 else np.nan,
            LMSD=float(np.nanmean(lmsd_list)) if len(lmsd_list) > 0 else np.nan,
            # VGGishDist=float(np.nanmean(vggish_list)),
            # FAD=float(np.nanmean(fad_list)),
            MRSTFT=float(np.nanmean(mrstft_list)),
        )


if __name__ == "__main__":
    print(
        "AudioCapsEvaluator (EventSep version).\n"
        "Use it from your training / benchmark script, e.g.:\n\n"
        "  evaluator = AudioCapsEvaluator(clapscore_type='audiosep')\n"
        "  results = evaluator(pl_audiosep, pl_flowsep)\n"
        "  print(results)\n"
    )
