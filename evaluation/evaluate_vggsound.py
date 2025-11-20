import os
import sys
import csv
import numpy as np
import torch
from tqdm import tqdm
import librosa
import lightning.pytorch as pl

# ===== AudioSep imports =====
sys.path.append(os.path.join(os.getcwd(), "AudioSep"))
from models.clap_encoder import CLAP_Encoder

# ===== PANNs imports =====
sys.path.append("./audioset_tagging_cnn/pytorch")
from audioset_tagging_cnn.pytorch.models import Cnn14_DecisionLevelAtt

from utils import calculate_sdr, calculate_sisdr


# ===== PANNs imports =====
sys.path.append("./audioset_tagging_cnn/pytorch")
from audioset_tagging_cnn.pytorch.models import Cnn14_DecisionLevelAtt


# =====================================================
# ========== Util: AudioSet 라벨 로드 ==================
# =====================================================
def load_audioset_labels():
    meta_path = "./audioset_tagging_cnn/metadata/class_labels_indices.csv"
    if not os.path.exists(meta_path):
        print(f"[WARN] metadata not found: {meta_path}")
        return None

    labels = []
    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["display_name"])

    if len(labels) != 527:
        print(f"[WARN] Expected 527 labels, got {len(labels)}")

    return labels


# =====================================================
# ========== Util: 프롬프트 → 클래스 id 매핑 ============
# =====================================================
def select_target_class(labels, text_prompt: str):
    """
    text_prompt (예: 'Lullaby')에 가장 잘 맞는 AudioSet 클래스 index를 찾는다.
    1) 정확히 같은 이름
    2) 부분 문자열 매칭
    3) 실패 시 'Music'으로 fallback
    """
    if labels is None:
        raise ValueError("AudioSet labels not loaded.")

    t = (text_prompt or "").lower()

    # 1) 완전 일치
    for i, name in enumerate(labels):
        if name.lower() == t:
            return i

    # 2) 부분 문자열
    for i, name in enumerate(labels):
        if t in name.lower():
            return i

    # 3) fallback: Music
    print(f"[WARN] Could not find class matching '{text_prompt}', using 'Music'.")
    try:
        return labels.index("Music")
    except ValueError:
        # 혹시 'Music'도 없으면 그냥 0번으로
        print("[WARN] 'Music' class not found in labels, fallback to class 0.")
        return 0


# =====================================================
# ========== Util: onset–offset 추출 ===================
# =====================================================
def extract_onset_offset(prob, threshold, frame_hop=0.01, min_frames=3):
    """
    prob: (T,) 특정 클래스의 framewise 확률
    threshold: 이 값 이상이면 활성으로 간주
    frame_hop: 한 프레임이 몇 초인지 (기본 0.01s)
    min_frames: 최소 이벤트 길이 (프레임 단위)

    returns: [(start_sec, end_sec), ...]
    """
    active = prob > threshold
    T = len(active)

    segments = []
    start = None

    for i in range(T):
        if active[i] and start is None:
            start = i
        if not active[i] and start is not None:
            end = i
            if end - start >= min_frames:
                segments.append((start * frame_hop, end * frame_hop))
            start = None

    if start is not None:
        end = T
        if end - start >= min_frames:
            segments.append((start * frame_hop, end * frame_hop))

    return segments


def interp_mask(mask, target_len):
    """
    mask: (T_panns,) binary or [0,1] mask
    target_len: waveform 길이 (samples)
    return: (target_len,) mask
    """
    T = len(mask)
    x1 = np.linspace(0, 1, T)
    x2 = np.linspace(0, 1, target_len)
    return np.interp(x2, x1, mask)


# =====================================================
# ========== VGGSound Evaluator ========================
# =====================================================
class VGGSoundEvaluator:
    def __init__(self, sampling_rate=32000, sed_threshold=0.3, use_sed_mask=True):
        self.sampling_rate = sampling_rate

        with open("evaluation/metadata/vggsound_eval.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            eval_list = [row for row in csv_reader][1:]

        self.eval_list = eval_list[:10]
        self.audio_dir = "evaluation/data/vggsound"

        # CLAP for CLAPScore
        print("Loading CLAP model for CLAPScore calculations ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clap = CLAP_Encoder().eval().to(self.device)

        # ===== PANNs (Cnn14_DecisionLevelAtt) for SED mask =====
        self.sed_threshold = sed_threshold
        self.use_sed_mask = use_sed_mask

        self.panns_device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = "./audioset_tagging_cnn/pytorch/Cnn14_DecisionLevelAtt_mAP=0.425.pth"
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"PANNs checkpoint not found at {ckpt}")

        print("Loading PANNs Cnn14_DecisionLevelAtt for SED-guided mask ...")
        self.panns = Cnn14_DecisionLevelAtt(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        state = torch.load(ckpt, map_location=self.panns_device)
        self.panns.load_state_dict(state["model"])
        self.panns = self.panns.to(self.panns_device).eval()

        # AudioSet class labels
        self.labels = load_audioset_labels()

    # --------------------------------------------------
    # 내부용: mixture + text_query → temporal mask
    # --------------------------------------------------
    def _get_temporal_mask(self, mixture: np.ndarray, text_query: str, device):
        """
        mixture: 1D numpy, 32kHz
        text_query: VGGSound 텍스트 라벨 (예: 'Lullaby music')
        return: torch.Tensor mask of shape (1, 1, T_waveform) or None
        """
        if not self.use_sed_mask:
            return None

        if self.labels is None:
            print("[WARN] AudioSet labels not loaded, disabling SED mask.")
            return None

        # target class id 선택
        cid = select_target_class(self.labels, text_query)
        target_name = self.labels[cid]

        # PANNs framewise inference
        wav_tensor = torch.tensor(mixture).float()[None].to(self.panns_device)
        with torch.no_grad():
            out = self.panns(wav_tensor, None)

        # (T, 527) → sigmoid 확률
        framewise = torch.sigmoid(out["framewise_output"][0]).cpu().numpy()
        target_prob = framewise[:, cid]  # (T,)

        # onset–offset 추출
        frame_hop = 320 / 32000.0  # 0.01s
        segments = extract_onset_offset(
            target_prob,
            threshold=self.sed_threshold,
            frame_hop=frame_hop,
            min_frames=3,
        )

        # binary mask (T_panns,)
        T = len(target_prob)
        mask = np.zeros(T)
        for s, e in segments:
            si = int(s / frame_hop)
            ei = int(e / frame_hop)
            mask[si:ei] = 1.0

        if mask.sum() == 0:
            # 디버깅/안정성을 위해, 아무 구간도 안 잡히면 gating 끔
            # (전체 1로 두고 AudioSep만에 맡긴다)
            mask[:] = 1.0

        # waveform 길이로 interpolation
        mask_interp = interp_mask(mask, len(mixture))
        mask_tensor = torch.tensor(mask_interp).float()[None, None].to(device)

        # 디버깅용 출력 (필요시)
        # print(f"[DEBUG] {target_name}: segments={segments}, active_ratio={mask_interp.mean():.3f}")

        return mask_tensor

    # --------------------------------------------------
    # 실제 평가 루프
    # --------------------------------------------------
    def __call__(self, pl_model: pl.LightningModule):
        print(f"Evaluation on VGGSound+ with [text label] queries (SED-guided masking={self.use_sed_mask}).")

        pl_model.eval()
        device = pl_model.device

        sdr_list = []
        sdri_list = []
        sisdr_list = []
        clap_score_list = []
        clapA_score_list = []

        with torch.no_grad():
            for eval_data in tqdm(self.eval_list):
                file_id, mix_wav, s0_wav, s0_text, s1_wav, s1_text = eval_data

                text_query = s0_text

                mixture_path = os.path.join(self.audio_dir, mix_wav)
                source_path = os.path.join(self.audio_dir, s0_wav)

                # Load audio
                source, _ = librosa.load(source_path, sr=self.sampling_rate, mono=True)
                mixture, _ = librosa.load(mixture_path, sr=self.sampling_rate, mono=True)

                # Compute baseline SDR
                sdr_no_sep = calculate_sdr(source, mixture)
                sisdr_no_sep = calculate_sisdr(source, mixture)

                # -----------------------------------------------------
                # SED-based temporal mask (EventSep 스타일)
                # -----------------------------------------------------
                temporal_mask = self._get_temporal_mask(mixture, text_query, device)

                mixture_tensor = torch.tensor(mixture)[None, None, :].float().to(device)
                if temporal_mask is not None:
                    mixture_tensor = mixture_tensor * temporal_mask

                # -----------------------------------------------------
                # AudioSep Text Embedding (condition)
                # -----------------------------------------------------
                cond = pl_model.query_encoder.get_query_embed(
                    modality="text",
                    text=[text_query],
                    device=device,
                )

                input_dict = {
                    "mixture": mixture_tensor,
                    "condition": cond,
                }

                # Separation
                sep = pl_model.ss_model(input_dict)["waveform"]
                sep = sep.squeeze().cpu().numpy()

                # Length alignment
                L = min(len(source), len(sep), len(mixture))
                source = source[:L]
                sep = sep[:L]
                mixture_trim = mixture[:L]

                # -----------------------------------------------------
                # SDR / SI-SDR
                sdr_sep = calculate_sdr(source, sep)
                sdri = sdr_sep - sdr_no_sep

                sisdr_sep = calculate_sisdr(source, sep)
                sisdri = sisdr_sep - sisdr_no_sep

                sdr_list.append(sdr_sep)
                sdri_list.append(sdri)
                sisdr_list.append(sisdr_sep)

                # -----------------------------------------------------
                # CLAPScore (text ↔ sep)
                # -----------------------------------------------------
                sep_tensor = torch.tensor(sep, dtype=torch.float32).unsqueeze(0).to(self.device)
                src_tensor = torch.tensor(source, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Audio embedding for separated output
                sep_emb = self.clap._get_audio_embed(sep_tensor)

                # Text embedding
                text_emb = self.clap._get_text_embed([text_query])

                # cosine similarity
                clap_score = torch.cosine_similarity(sep_emb, text_emb).item()
                clap_score_list.append(clap_score)

                # -----------------------------------------------------
                # CLAPScoreA (source ↔ sep)
                # -----------------------------------------------------
                src_emb = self.clap._get_audio_embed(src_tensor)

                clapA = torch.cosine_similarity(sep_emb, src_emb).item()
                clapA_score_list.append(clapA)

        # -----------------------------------------------------
        # Final statistics
        # -----------------------------------------------------
        return {
            "SDR": float(np.mean(sdr_list)),
            "SDRi": float(np.mean(sdri_list)),
            "SISDR": float(np.mean(sisdr_list)),
            "CLAPScore": float(np.mean(clap_score_list)),
            "CLAPScoreA": float(np.mean(clapA_score_list)),
        }
