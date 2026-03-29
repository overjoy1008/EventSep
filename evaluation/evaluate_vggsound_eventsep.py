# ====================== evaluate_vggsound_eventsep.py ======================

import os
import sys
import csv
import random
import subprocess
from pathlib import Path

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


# ============================================================================ 
#                 EventSep Evaluator (VGGSound + CLAPScore 선택형)
# ============================================================================ 

class VGGSoundEventSepEvaluator:
    """
    EventSep용 VGGSound 평가기.

    - PANNs(Cnn14_DecisionLevelAtt)를 이용한 SED-guided masking
    - AudioSep + FlowSep 듀얼 분리 후 ensemble
    - 선택적으로 Demucs 전단 분리(use_demucs)
    - CLAPScore backbone 선택 가능:
        * "audiosep"                      → AudioSep 기본 CLAP_Encoder
        * "630k_best"                     → laion_clap 630k-best
        * "630k_audioset_best"           → laion_clap 630k-audioset-best
        * "630k_fusion_best"             → laion_clap 630k-fusion-best
        * "630k_audioset_fusion_best"    → laion_clap 630k-audioset-fusion-best
        * "music_speech_audioset"        → AudioSep 제공 music_speech_audioset_epoch_15_esc_89.98.pt
        * "music_speech"                 → AudioSep 제공 music_speech_epoch_15_esc_89.25.pt
    """

    def __init__(
        self,
        sampling_rate: int = 32000,
        sed_threshold: float = 0.4,
        mask_mode: str = "soft",          # "soft" / "hard" / "none"
        ensemble: str = "audiosep",       # "audiosep" / "flowsep" / "ensemble_a" / "ensemble_b" ...
        ensemble_rate: float = 0.3,
        ensemble_freq: int = 4000,
        clapscore_type: str = "audiosep", # 위 설명 참조
        vgg_csv_path: str = "evaluation/metadata/vggsound_eval.csv",
        vgg_audio_dir: str = "evaluation/data/vggsound",
        debug_num_samples: int = 5,
        debug_dir: str = "evaluation/debug/eventsep",
        embedding_model_type: str = "minilm",   # AMTH
        demo_mode: bool = False,
        use_demucs: bool = False,
        demucs_root: str = "demucs",
        demucs_model: str = "htdemucs",
    ):
        self.sr = sampling_rate
        self.sed_threshold = sed_threshold
        self.mask_mode = mask_mode
        self.ensemble = ensemble
        self.ensemble_rate = ensemble_rate
        self.ensemble_freq = ensemble_freq
        self.clapscore_type = clapscore_type
        self.embedding_model_type = embedding_model_type  # AMTH
        self.demo_mode = demo_mode

        # ----------------- Demucs 관련 설정 -----------------
        self.use_demucs = use_demucs
        self.demucs_root = demucs_root
        self.demucs_model = demucs_model

        # ----------------- VGGSound 메타 -----------------
        with open(vgg_csv_path, "r") as f:
            rows = [row for row in csv.reader(f)][1:]  # header 제거
            
        if self.demo_mode:
            self.eval_list = rows[:10]
        else:
            self.eval_list = rows
        
        self.audio_dir = vgg_audio_dir

        # ----------------- Device -----------------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # =================================================
        #              Separation용 AudioSep CLAP
        # =================================================
        # (AudioSep의 query_encoder / ss_model과는 별개로,
        #  CLAP backbone만 여기서도 한 번 사용)
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
            self.clap_score_text, self.clap_score_audio = load_auto_clap_model(device=self.device)
            self.clap_score = None   # 사용하지 않음
        else:
            self.clap_score = load_clapscore_model(model_type=self.clapscore_type, device=self.device)
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
            self.debug_samples = set([row[0] for row in selected])

        print(f"[EventSep] Using CLAPScore model: {self.clapscore_type}")
        print(f"[EventSep] VGGSound eval pairs: {len(self.eval_list)}")
        print(f"[EventSep] Debug samples: {len(self.debug_samples)} will be saved in '{self.debug_dir}'")
        if self.use_demucs:
            print(f"[EventSep] Demucs semantic routing ENABLED (root={self.demucs_root}, model={self.demucs_model})")

    # =====================================================================
    #                        Demucs helper
    # =====================================================================

    def _run_demucs_stem(self, mix_path: str, target: str) -> np.ndarray:
        """
        Demucs 안전 실행:
        - 원본 mix_path를 tmp 파일로 복사 (파일명 이슈 해결)
        - Demucs 실행 후 tmp_output 폴더에서 stem 추출
        """
        import uuid
        import shutil

        # 1. 안전한 임시 파일명 생성
        tmp_id = str(uuid.uuid4())[:8]
        tmp_in = f"/tmp/eventsep_demucs_{tmp_id}.wav"
        tmp_out = f"/tmp/eventsep_demucs_out_{tmp_id}"

        os.makedirs(tmp_out, exist_ok=True)

        # 2. mix_path → tmp_in 으로 안전 복사
        shutil.copyfile(mix_path, tmp_in)

        # 3. Demucs 실행
        cmd = [
            "python3", "-m", "demucs.separate",
            "-n", self.demucs_model,
            "--device", "cuda" if self.device == "cuda" else "cpu",
            "-o", tmp_out,
            tmp_in,
        ]

        # 항상 Demucs repo 내부에서 실행
        subprocess.run(cmd, check=True, cwd=self.demucs_root)

        # 4. Demucs 출력 경로 추적
        # tmp_in 파일명의 basename은 eventsep_demucs_<id>
        base = Path(tmp_in).stem  # eventsep_demucs_xxxx
        stem_dir = os.path.join(tmp_out, self.demucs_model, base)
        stem_path = os.path.join(stem_dir, f"{target}.wav")

        if not os.path.exists(stem_path):
            raise FileNotFoundError(f"[Demucs] Stem file not found (checked): {stem_path}")

        wav, sr = librosa.load(stem_path, sr=self.sr, mono=True)

        # 5. 임시 파일 정리(optional)
        # shutil.rmtree(tmp_out)
        # os.remove(tmp_in)

        return wav.astype(np.float32)

    # =====================================================================
    #                           Evaluation Loop
    # =====================================================================

    def __call__(self, pl_audiosep, pl_flowsep):
        """
        pl_audiosep : AudioSep LightningModule (query_encoder / ss_model 포함)
        pl_flowsep  : FlowSep LightningModule (ss_model에서 'waveform', 'random_start' 반환)
        """

        sdr_list, sdri_list, sisdr_list = [], [], []
        clap_list, clapA_list = [], []

        # ===== 추가 지표 리스트 =====
        stoi_list, estoi_list = [], []
        pesq_list, lmsd_list = [], []
        vggish_list, fad_list, mrstft_list = [], [], []

        pl_audiosep.eval()
        pl_flowsep.eval()

        with torch.no_grad():
            for row in tqdm(self.eval_list):
                # CSV columns: file_id, mix_wav, s0_wav, s0_text, ...
                file_id, mix_wav, s0_wav, s0_text, _, _ = row
                text_query = s0_text

                # ----------------- 파일 로드 -----------------
                mix_path = os.path.join(self.audio_dir, mix_wav)
                src_path = os.path.join(self.audio_dir, s0_wav)

                source, _ = librosa.load(src_path, sr=self.sr, mono=True)
                mixture, _ = librosa.load(mix_path, sr=self.sr, mono=True)

                # 길이 맞추기 (기본 정렬)
                L0 = min(len(source), len(mixture))
                source = source[:L0]
                mixture = mixture[:L0]

                # no-separation SDR (baseline)
                sdr_no_sep = calculate_sdr(source, mixture)

                # ----------------- Demucs semantic routing -----------------
                demucs_sep = None
                demucs_target = None

                # Prompt → AudioSet label / Demucs target 추론
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
                        demucs_sep = self._run_demucs_stem(mix_path, demucs_target)
                        # 길이 정렬
                        Ld = min(len(demucs_sep), len(mixture), len(source))
                        demucs_sep = demucs_sep[:Ld]
                        mixture = mixture[:Ld]
                        source = source[:Ld]

                        # 나머지 others = mixture - demucs_target
                        mixture_for_eventsep = mixture - demucs_sep
                        print(f"[Demucs] Routed prompt '{text_query}' → stem '{demucs_target}'")
                    except Exception as e:
                        print(f"[Demucs] Failed to run Demucs for '{text_query}': {e}")
                        demucs_sep = None
                        mixture_for_eventsep = mixture

                # ----------------- SED-guided masking -----------------
                if self.mask_mode != "none":
                    # Demucs가 켜진 경우 cid는 dict일 수 있으므로,
                    # SED는 사용할 수 있는 경우에만 사용.
                    cid_sed = cid
                    if isinstance(cid_sed, dict):
                        cid_sed = None  # Demucs dict이면 SED는 비활성화

                    if cid_sed is None:
                        # SED index가 없으면 masking 없이 통과
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

                # [B, 1, T]
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
                sep_a = sep_a.squeeze().detach().cpu().numpy()  # [T]

                # =====================================================
                #                     FlowSep 추론
                # =====================================================
                out_f = pl_flowsep.ss_model({"mixture": mix_tensor, "text": text_query})
                sep_f16 = out_f["waveform"].squeeze().detach().cpu().numpy()  # [T_16k]
                random_start16 = int(out_f["random_start"])

                # 16k → 32k
                sep_f = librosa.resample(sep_f16, orig_sr=16000, target_sr=self.sr)
                Lf = len(sep_f)

                # random_start를 32k 타임스텝으로 변환
                start32 = int(random_start16 * (self.sr / 16000.0))

                # AudioSep 길이 보정 (필요시 padding)
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

                # 최종 sep: AudioSep 전체 + 특정 구간만 ensemble 결과로 교체
                sep = np.copy(sep_a)
                sep[start32:start32 + len(seg_e)] = seg_e

                # ----------------- Demucs residual fusion -----------------
                if demucs_sep is not None:
                    # EventSep 분리는 "others"에 대해서만 수행됐으므로,
                    # 최종 타겟은 Demucs stem + EventSep residual 로 구성
                    L = min(len(source), len(sep), len(mixture), len(demucs_sep))
                    source = source[:L]
                    mixture = mixture[:L]
                    sep = sep[:L]
                    demucs_sep = demucs_sep[:L]
                    sep = sep + demucs_sep
                else:
                    # 길이 맞추기 (기존 경로)
                    L = min(len(source), len(sep), len(mixture))
                    source = source[:L]
                    sep = sep[:L]
                    mixture = mixture[:L]

                # =====================================================
                #                         Metrics
                # =====================================================

                # --- SDR / SDRi ---
                sdr_sep = calculate_sdr(source, sep)
                sdri_list.append(sdr_sep - sdr_no_sep)
                sdr_list.append(sdr_sep)

                # --- SI-SDR ---
                sisdr_sep = calculate_sisdr(source, sep)
                sisdr_list.append(sisdr_sep)

                # --- 추가: STOI / ESTOI / PESQ / LMSD ---
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

                # =====================================================
                #               AUTO MODE (dual CLAP)
                # =====================================================
                if self.clapscore_type == "auto":

                    # ----- CLAPScore (text ↔ sep) using fusion_best -----
                    sep_emb_text = self.clap_score_text.get_audio_embedding_from_data(
                        sep_t, use_tensor=True
                    )
                    text_emb = self.clap_score_text.get_text_embedding(
                        [text_query], use_tensor=True
                    )
                    clap_val = torch.nn.functional.cosine_similarity(
                        sep_emb_text, text_emb
                    ).item()

                    # ----- CLAPScoreA (source ↔ sep) using audio_best -----
                    sep_emb_aud = self.clap_score_audio.get_audio_embedding_from_data(
                        sep_t, use_tensor=True
                    )
                    src_emb_aud = self.clap_score_audio.get_audio_embedding_from_data(
                        src_t, use_tensor=True
                    )
                    clapA_val = torch.nn.functional.cosine_similarity(
                        sep_emb_aud, src_emb_aud
                    ).item()

                # =====================================================
                #               NON-AUTO MODE (original)
                # =====================================================
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

                # =====================================================
                #                         Debug Save
                # =====================================================
                if file_id in self.debug_samples:
                    base = os.path.join(self.debug_dir, file_id)
                    os.makedirs(os.path.dirname(base), exist_ok=True)

                    sf.write(base + "_mix.wav", mixture, self.sr)
                    sf.write(base + "_src.wav", source, self.sr)
                    sf.write(base + "_sep.wav", sep, self.sr)
                    sf.write(base + "_audiosep.wav", sep_a[:L], self.sr)
                    sf.write(base + "_flowsep.wav", sep_f[: min(Lf, L)], self.sr)
                    if demucs_sep is not None:
                        sf.write(base + "_demucs.wav", demucs_sep[:L], self.sr)

        # ----------------- 평균 결과 반환 -----------------
        return dict(
            SDR=np.mean(sdr_list),
            SDRi=np.mean(sdri_list),
            SISDR=np.mean(sisdr_list),
            CLAPScore=np.mean(clap_list),
            CLAPScoreA=np.mean(clapA_list),

            # ===== 추가 지표 평균 =====
            STOI=np.nanmean(stoi_list) if len(stoi_list) > 0 else np.nan,
            ESTOI=np.nanmean(estoi_list) if len(estoi_list) > 0 else np.nan,
            PESQ=np.nanmean(pesq_list) if len(pesq_list) > 0 else np.nan,
            LMSD=np.nanmean(lmsd_list) if len(lmsd_list) > 0 else np.nan,

            # 추가된 지표들
            # VGGishDist=np.nanmean(vggish_list),
            # FAD=np.nanmean(fad_list),
            MRSTFT=np.nanmean(mrstft_list),
        )


if __name__ == "__main__":
    print(
        "This file defines VGGSoundEventSepEvaluator.\n"
        "Use it from your training / benchmark script, e.g.:\n\n"
        "  evaluator = VGGSoundEventSepEvaluator(clapscore_type='audiosep')\n"
        "  results = evaluator(pl_audiosep, pl_flowsep)\n"
        "  print(results)\n"
    )
