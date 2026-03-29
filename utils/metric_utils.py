"""
추가 평가 지표 모음 (EventSep 전용):

- STOI / ESTOI         : pystoi 필요
- PESQ                 : pesq 필요
- LMSD                 : log-mel spectral distance
- FAD                  : Fréchet Audio Distance (vggish 필요)
- VGGish Distance      : ref/est embedding L2
- MR-STFT Distance     : multi-resolution STFT spectral distance

모듈 미설치/오류 시 NaN 반환.
"""

import numpy as np
import warnings
import librosa

# Optional deps
_HAS_PYSTOI = False
_HAS_PESQ = False
_HAS_VGGISH = False

try:
    from pystoi import stoi as _stoi
    _HAS_PYSTOI = True
except Exception:
    pass

try:
    from pesq import pesq as _pesq
    _HAS_PESQ = True
except Exception:
    pass

# ---- VGGish Embedding Loader ----
try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from vggish import VGGish  # pip install torchvggish
    _HAS_VGGISH = True
except Exception:
    _HAS_VGGISH = False

# warnings only once
_warned = set()

def _warn_once(msg):
    if msg not in _warned:
        warnings.warn(msg)
        _warned.add(msg)



# =====================================================
# STOI / ESTOI
# =====================================================

def compute_stoi(ref, est, sr):
    if not _HAS_PYSTOI:
        _warn_once("[WARN] pystoi not installed → STOI = NaN")
        return np.nan

    L = min(len(ref), len(est))
    if L == 0:
        return np.nan

    try:
        return float(_stoi(ref[:L], est[:L], sr, extended=False))
    except Exception:
        return np.nan


def compute_estoi(ref, est, sr):
    if not _HAS_PYSTOI:
        _warn_once("[WARN] pystoi not installed → ESTOI = NaN")
        return np.nan

    L = min(len(ref), len(est))
    if L == 0:
        return np.nan

    try:
        return float(_stoi(ref[:L], est[:L], sr, extended=True))
    except Exception:
        return np.nan



# =====================================================
# PESQ
# =====================================================

def compute_pesq(ref, est, sr):
    """
    PESQ (Perceptual Evaluation of Speech Quality) 계산.

    - sr == 8000  → 'nb'
    - sr == 16000 → 'wb'
    나머지 샘플레이트는 librosa로 16k로 리샘플 후 'wb'로 계산.
    """
    global _pesq_warned

    if not _HAS_PESQ:
        if not _pesq_warned:
            warnings.warn(
                "[WARN] pesq 패키지가 설치되어 있지 않아 PESQ를 계산할 수 없습니다. "
                "pip install pesq 로 설치 후 사용하세요."
            )
            _pesq_warned = True
        return np.nan

    if len(ref) == 0 or len(est) == 0:
        return np.nan

    L = min(len(ref), len(est))
    ref = ref[:L]
    est = est[:L]

    # PESQ는 반드시 16k 또는 8k만 지원
    target_sr = 16000
    if sr != target_sr:
        ref = librosa.resample(ref, orig_sr=sr, target_sr=target_sr)
        est = librosa.resample(est, orig_sr=sr, target_sr=target_sr)

    try:
        return float(_pesq(target_sr, ref, est, "wb"))
    except Exception:
        return np.nan



# =====================================================
# LMSD (Log-Mel Spectral Distance)
# =====================================================

def compute_lmsd(ref, est, sr, n_fft=1024, hop=256, n_mels=64):
    L = min(len(ref), len(est))
    if L == 0:
        return np.nan

    # librosa >= 0.10: 모든 인자는 keyword-only
    mel_ref = librosa.feature.melspectrogram(
        y=ref[:L],
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
    )
    mel_est = librosa.feature.melspectrogram(
        y=est[:L],
        sr=sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
    )

    log_ref = librosa.power_to_db(mel_ref)
    log_est = librosa.power_to_db(mel_est)

    T = min(log_ref.shape[1], log_est.shape[1])
    diff = log_ref[:, :T] - log_est[:, :T]
    return float(np.sqrt(np.mean(diff ** 2)))




# =====================================================
# VGGish Embedding Distance
# =====================================================

_vgg_model = None

def _get_vggish():
    global _vgg_model
    if _vgg_model is None:
        if not _HAS_VGGISH:
            raise RuntimeError("torchvggish not installed")
        _vgg_model = VGGish(pretrained=True).eval()
    return _vgg_model


def compute_vggish_distance(ref, est, sr):
    """
    VGGish feature L2 distance (작을수록 좋음)
    """
    if not _HAS_VGGISH:
        _warn_once("[WARN] VGGish not available → VGGishDist = NaN")
        return np.nan

    L = min(len(ref), len(est))
    if L == 0:
        return np.nan

    try:
        model = _get_vggish()
        # convert to 16k mono
        target_sr = 16000
        if sr != target_sr:
            ref = librosa.resample(ref[:L], orig_sr=sr, target_sr=target_sr)
            est = librosa.resample(est[:L], orig_sr=sr, target_sr=target_sr)
        else:
            ref = ref[:L]
            est = est[:L]

        with torch.no_grad():
            r = torch.tensor(ref).float().unsqueeze(0)
            e = torch.tensor(est).float().unsqueeze(0)
            fr = model(r).cpu().numpy().squeeze()
            fe = model(e).cpu().numpy().squeeze()
        return float(np.linalg.norm(fr - fe))
    except Exception as e:
        _warn_once(f"[WARN] VGGishDist error: {e}")
        return np.nan



# =====================================================
# FAD (Fréchet Audio Distance)
# =====================================================

def compute_fad(ref, est, sr):
    """
    FAD = Fréchet Distance between VGGish embeddings stats:
    d^2 = ||mu1 - mu2||^2 + Tr(Σ1 + Σ2 - 2(Σ1 Σ2)^(1/2))

    여기서는 ref/est 각각 하나의 샘플만 있으므로
    cov = 0 으로 단순화된 FAD-like distance 계산.
    """
    if not _HAS_VGGISH:
        _warn_once("[WARN] FAD requires VGGish → FAD = NaN")
        return np.nan

    # reuse VGGishDistance as per-sample
    d = compute_vggish_distance(ref, est, sr)
    return d  # 단일 샘플에서는 VGGishDistance와 동일하게 처리



# =====================================================
# MR-STFT Distance (spectral convergence)
# =====================================================

def compute_mrstft(ref, est, sr, n_fft_list=[1024, 2048], hop_list=[256, 512]):
    """
    Multi-resolution STFT spectral convergence:
    SC = |||X|-|Y||| / ||X||
    """
    L = min(len(ref), len(est))
    if L == 0:
        return np.nan

    ref = ref[:L]
    est = est[:L]

    sc_list = []
    for nfft, hop in zip(n_fft_list, hop_list):
        X = librosa.stft(ref, n_fft=nfft, hop_length=hop)
        Y = librosa.stft(est, n_fft=nfft, hop_length=hop)

        magX = np.abs(X)
        magY = np.abs(Y)

        num = np.linalg.norm(magX - magY)
        den = np.linalg.norm(magX) + 1e-9
        sc = num / den
        sc_list.append(sc)

    return float(np.mean(sc_list))
