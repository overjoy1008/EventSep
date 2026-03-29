# ====================== evaluate_audioset_eventsep.py ======================

import os
import sys
import re
import random
from typing import Dict, List

import pandas as pd
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
from utils.metric_utils import (
    compute_stoi,
    compute_estoi,
    compute_pesq,
    compute_lmsd,
    compute_vggish_distance,
    compute_fad,
    compute_mrstft,
)

# ===== AudioSet meta (원래 코드 유지) =====
meta_csv_file = "evaluation/metadata/class_labels_indices.csv"
df = pd.read_csv(meta_csv_file, sep=",")

IDS = df["mid"].tolist()
LABELS = df["display_name"].tolist()
CLASSES_NUM = len(LABELS)

IX_TO_LB = {i: label for i, label in enumerate(LABELS)}


class AudioSetEvaluator:
    """
    EventSep용 AudioSet 평가기.

    - class_id별 디렉토리 구조는 기존 AudioSetEvaluator와 동일
    - VGGSoundEventSepEvaluator 구조로 통합 (SED, Demucs, ensemble, CLAPScore 등)
    """

    def __init__(
        self,
        audios_dir: str = "evaluation/data/audioset",
        classes_num: int = 527,
        sampling_rate: int = 32000,
        number_per_class: int = 10,
        sed_threshold: float = 0.4,
        mask_mode: str = "soft",
        ensemble: str = "audiosep",
        ensemble_rate: float = 0.3,
        ensemble_freq: int = 4000,
        clapscore_type: str = "audiosep",
        debug_num_samples: int = 5,
        debug_dir: str = "evaluation/debug/audioset_eventsep",
        embedding_model_type: str = "minilm",
        demo_mode: bool = False,
        use_demucs: bool = False,
        demucs_root: str = "demucs",
        demucs_model: str = "htdemucs",
    ) -> None:
        self.audios_dir = audios_dir
        self.classes_num = classes_num
        self.number_per_class = number_per_class
        self.sr = sampling_rate
        self.sed_threshold = sed_threshold
        self.mask_mode = mask_mode
        self.ensemble = ensemble
        self.ensemble_rate = ensemble_rate
        self.ensemble_freq = ensemble_freq
        self.clapscore_type = clapscore_type
        self.embedding_model_type = embedding_model_type
        self.demo_mode = demo_mode

        self.use_demucs = use_demucs
        self.demucs_root = demucs_root
        self.demucs_model = demucs_model

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Separation용 CLAP
        self.clap_sep = CLAP_Encoder().eval().to(self.device)

        # PANNs
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

        self.labels = load_audioset_labels()

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

        # Debug
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)

        # AudioSet은 전체 샘플 개수를 알기 어렵기 때문에,
        # 디버그 샘플은 (class_id, audio_name) 조합 중 일부만 랜덤 선택
        self.debug_samples = set()
        if debug_num_samples > 0:
            # 대충 앞에서부터 class를 돌면서 구성
            tmp_list = []
            for class_id in range(self.classes_num):
                sub_dir = os.path.join(self.audios_dir, f"class_id={class_id}")
                if not os.path.isdir(sub_dir):
                    continue
                names = self._get_audio_names(sub_dir)
                for n in names:
                    tmp_list.append((class_id, n))
            if len(tmp_list) > debug_num_samples:
                tmp_list = random.sample(tmp_list, debug_num_samples)
            self.debug_samples = set([f"{cid}_{name}" for cid, name in tmp_list])

        print(f"[AudioSet-EventSep] Using CLAPScore model: {self.clapscore_type}")
        print(f"[AudioSet-EventSep] Classes: {self.classes_num}")
        print(f"[AudioSet-EventSep] Debug samples: {len(self.debug_samples)} "
              f"will be saved in '{self.debug_dir}'")
        if self.use_demucs:
            print(f"[AudioSet-EventSep] Demucs semantic routing ENABLED "
                  f"(root={self.demucs_root}, model={self.demucs_model})")

    # =====================================================================
    #                        Demucs helper
    # =====================================================================

    def _run_demucs_stem(self, mix_path: str, target: str) -> np.ndarray:
        import uuid
        import shutil

        tmp_id = str(uuid.uuid4())[:8]
        tmp_in = f"/tmp/audioset_demucs_{tmp_id}.wav"
        tmp_out = f"/tmp/audioset_demucs_out_{tmp_id}"

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

        base = os.path.basename(tmp_in).split(".")[0]
        stem_dir = os.path.join(tmp_out, self.demucs_model, base)
        stem_path = os.path.join(stem_dir, f"{target}.wav")

        if not os.path.exists(stem_path):
            raise FileNotFoundError(f"[Demucs] Stem file not found: {stem_path}")

        wav, sr = librosa.load(stem_path, sr=self.sr, mono=True)
        return wav.astype(np.float32)

    # =====================================================================
    #                        Evaluation Loop
    # =====================================================================

    @torch.no_grad()
    def __call__(self, pl_audiosep: pl.LightningModule, pl_flowsep: pl.LightningModule) -> Dict:
        pl_audiosep.eval()
        pl_flowsep.eval()

        sdr_list, sdri_list, sisdr_list = [], [], []
        clap_score_list, clapA_score_list = [], []

        stoi_list, estoi_list = [], []
        pesq_list, lmsd_list = [], []
        vggish_list, fad_list, mrstft_list = [], [], []

        print("Evaluation on AudioSet with [text label] queries (EventSep).")

        for class_id in tqdm(range(self.classes_num)):
            sub_dir = os.path.join(self.audios_dir, f"class_id={class_id}")
            if not os.path.isdir(sub_dir):
                continue

            audio_names = self._get_audio_names(audios_dir=sub_dir)

            for audio_index, audio_name in enumerate(audio_names):
                if audio_index == self.number_per_class:
                    break

                source_path = os.path.join(sub_dir, f"{audio_name},source.wav")
                mixture_path = os.path.join(sub_dir, f"{audio_name},mixture.wav")

                if not (os.path.exists(source_path) and os.path.exists(mixture_path)):
                    continue

                source, _ = librosa.load(source_path, sr=self.sr, mono=True)
                mixture, _ = librosa.load(mixture_path, sr=self.sr, mono=True)

                L0 = min(len(source), len(mixture))
                source = source[:L0]
                mixture = mixture[:L0]

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)

                text_query = IX_TO_LB[class_id]

                # Demucs routing
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

                # SED-guided masking
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

                # AudioSep
                cond = pl_audiosep.query_encoder.get_query_embed(
                    modality="text",
                    text=[text_query],
                    device=self.device,
                )
                sep_a = pl_audiosep.ss_model(
                    {"mixture": mix_tensor, "condition": cond}
                )["waveform"]
                sep_a = sep_a.squeeze().detach().cpu().numpy()

                # FlowSep
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

                # Metrics
                sdr_sep = calculate_sdr(ref=source, est=sep)
                sdri = sdr_sep - sdr_no_sep
                sisdr_sep = calculate_sisdr(ref=source, est=sep)

                sdr_list.append(sdr_sep)
                sdri_list.append(sdri)
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

                # CLAPScore
                sep_tensor = torch.tensor(sep, dtype=torch.float32).unsqueeze(0).to(
                    self.device
                )
                src_tensor = torch.tensor(source, dtype=torch.float32).unsqueeze(0).to(
                    self.device
                )

                if self.clapscore_type == "auto":
                    sep_emb_text = self.clap_score_text.get_audio_embedding_from_data(
                        sep_tensor, use_tensor=True
                    )
                    text_emb = self.clap_score_text.get_text_embedding(
                        [text_query], use_tensor=True
                    )
                    clap_score = torch.cosine_similarity(sep_emb_text, text_emb).item()

                    sep_emb_aud = self.clap_score_audio.get_audio_embedding_from_data(
                        sep_tensor, use_tensor=True
                    )
                    src_emb_aud = self.clap_score_audio.get_audio_embedding_from_data(
                        src_tensor, use_tensor=True
                    )
                    clapA = torch.cosine_similarity(sep_emb_aud, src_emb_aud).item()
                else:
                    if isinstance(self.clap_score, CLAP_Encoder):
                        sep_emb = self.clap_score._get_audio_embed(sep_tensor)
                        src_emb = self.clap_score._get_audio_embed(src_tensor)
                        text_emb = self.clap_score._get_text_embed([text_query])
                    else:
                        sep_emb = self.clap_score.get_audio_embedding_from_data(
                            sep_tensor, use_tensor=True
                        )
                        src_emb = self.clap_score.get_audio_embedding_from_data(
                            src_tensor, use_tensor=True
                        )
                        text_emb = self.clap_score.get_text_embedding(
                            [text_query], use_tensor=True
                        )

                    clap_score = torch.cosine_similarity(sep_emb, text_emb).item()
                    clapA = torch.cosine_similarity(sep_emb, src_emb).item()

                clap_score_list.append(clap_score)
                clapA_score_list.append(clapA)

                # Debug
                debug_key = f"{class_id}_{audio_name}"
                if debug_key in self.debug_samples:
                    base = os.path.join(self.debug_dir, debug_key)
                    os.makedirs(os.path.dirname(base), exist_ok=True)
                    sf.write(base + "_mix.wav", mixture, self.sr)
                    sf.write(base + "_src.wav", source, self.sr)
                    sf.write(base + "_sep.wav", sep, self.sr)
                    sf.write(base + "_audiosep.wav", sep_a[:L], self.sr)
                    sf.write(base + "_flowsep.wav", sep_f[: min(Lf, L)], self.sr)
                    if demucs_sep is not None:
                        sf.write(base + "_demucs.wav", demucs_sep[:L], self.sr)

        return {
            "SDR": float(np.mean(sdr_list)),
            "SDRi": float(np.mean(sdri_list)),
            "SISDR": float(np.mean(sisdr_list)),
            "CLAPScore": float(np.mean(clap_score_list)),
            "CLAPScoreA": float(np.mean(clapA_score_list)),
            "STOI": float(np.nanmean(stoi_list)) if len(stoi_list) > 0 else np.nan,
            "ESTOI": float(np.nanmean(estoi_list)) if len(estoi_list) > 0 else np.nan,
            "PESQ": float(np.nanmean(pesq_list)) if len(pesq_list) > 0 else np.nan,
            "LMSD": float(np.nanmean(lmsd_list)) if len(lmsd_list) > 0 else np.nan,
            # "VGGishDist": float(np.nanmean(vggish_list)),
            # "FAD": float(np.nanmean(fad_list)),
            "MRSTFT": float(np.nanmean(mrstft_list)),
        }

    def _get_audio_names(self, audios_dir: str) -> List[str]:
        audio_names = sorted(os.listdir(audios_dir))
        audio_names = [x for x in audio_names if ".wav" in x]
        audio_names = [
            re.search(r"(.*),(mixture|source)\.wav", x).group(1) for x in audio_names
        ]
        return sorted(list(set(audio_names)))


if __name__ == "__main__":
    print(
        "AudioSetEvaluator (EventSep version).\n"
        "  evaluator = AudioSetEvaluator(clapscore_type='audiosep')\n"
        "  results = evaluator(pl_audiosep, pl_flowsep)\n"
        "  print(results)\n"
    )
