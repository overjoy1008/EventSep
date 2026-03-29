import os
import sys
import torch
import numpy as np
import argparse
import librosa
import csv

# AudioSep 경로를 sys.path 최상단에 추가
sys.path.insert(0, "./AudioSep")  # << 변경된 부분
sys.path.append("./AudioSep")

# ===== NEW: visualization utils =====
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa.display


def save_waveform(path, audio, sr):
    import soundfile as sf

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio, sr)


def save_spectrogram(path, audio, sr):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.figure(figsize=(10, 4))
    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=512, hop_length=128)), ref=np.max
    )
    librosa.display.specshow(S, sr=sr, hop_length=128, y_axis="linear", x_axis="time")
    plt.xlim(0, 10)
    plt.ylim(0, 8000)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =====================================================
# ===== First load AudioSep & PANNs  ==================
# =====================================================
sys.path.append("./AudioSep")
from pipeline import build_audiosep

sys.path.append("./audioset_tagging_cnn/pytorch")
from audioset_tagging_cnn.pytorch.models import Cnn14_DecisionLevelAtt

# =====================================================
# ===== sentence-transformers =========================
# =====================================================
from sentence_transformers import SentenceTransformer

_TEXT_EMB_MODEL = None
_LABEL_TEXTS = None
_LABEL_EMB = None


# =====================================================
# ================== Utility ==========================
# =====================================================
def load_audio(path, sr=32000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


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
    return labels


# ========== Embeddings ==========
def get_text_embedding_model(debug=False):
    global _TEXT_EMB_MODEL
    if _TEXT_EMB_MODEL is None:
        _TEXT_EMB_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _TEXT_EMB_MODEL


def build_label_embeddings(labels, debug=False):
    global _LABEL_TEXTS, _LABEL_EMB
    if labels is None:
        return None, None

    if _LABEL_TEXTS is not None:
        return _LABEL_TEXTS, _LABEL_EMB

    model = get_text_embedding_model(debug)
    _LABEL_TEXTS = labels
    _LABEL_EMB = model.encode(labels, normalize_embeddings=True)
    return _LABEL_TEXTS, _LABEL_EMB


def select_target_class(labels, text_prompt, debug=False):
    q = text_prompt.lower().strip()

    # exact match
    for i, name in enumerate(labels):
        if name.lower() == q:
            return i
    # partial match
    for i, name in enumerate(labels):
        if q in name.lower() or name.lower() in q:
            return i

    # embedding fallback
    label_texts, label_emb = build_label_embeddings(labels, debug)
    model = get_text_embedding_model(debug)
    q_emb = model.encode([text_prompt], normalize_embeddings=True)[0]
    sim = np.dot(label_emb, q_emb)
    return int(np.argmax(sim))


# ========== SED ==========
def interp_mask(mask, target_len):
    x1 = np.linspace(0, 1, len(mask))
    x2 = np.linspace(0, 1, target_len)
    return np.interp(x2, x1, mask)


def run_panns_sed_mask(
    audio, device, text_prompt, threshold=0.3, masking="hard", debug=False, vis_dir=None
):
    """
    Return:
        prob: (num_frames,)
        mask: (num_frames,)
    """

    # ============ return identity mask when masking disabled ============
    if masking == "none":
        return None, np.ones_like(audio)

    # ============ load PANNs model ============
    ckpt = "./audioset_tagging_cnn/pytorch/Cnn14_DecisionLevelAtt_mAP=0.425.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(ckpt)

    model = Cnn14_DecisionLevelAtt(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527,
    )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    # ============ label + target class ============
    labels = load_audioset_labels()
    class_id = select_target_class(labels, text_prompt, debug)

    # prepare audio
    wav_tensor = torch.tensor(audio).float()[None].to(device)

    # ============ forward ============
    with torch.no_grad():
        out = model(wav_tensor, None)

    framewise = torch.sigmoid(out["framewise_output"][0]).cpu().numpy()  # (T, 527)
    prob = framewise[:, class_id]  # (T,)

    # ============ create mask ============
    if masking == "hard":
        mask = (prob >= threshold).astype(np.float32)

    elif masking == "soft":
        mask = np.minimum(prob / max(threshold, 1e-6), 1.0).astype(np.float32)

    elif masking == "soft-rel":
        p_min = prob.min()
        p_max = prob.max()

        if p_max - p_min < 1e-12:
            mask = np.ones_like(prob, dtype=np.float32)
        else:
            mask = (prob - p_min) / (p_max - p_min)
            mask = mask.astype(np.float32)

    elif masking == "soft-fade":
        # Step 1: relative soft mask
        p_min = prob.min()
        p_max = prob.max()
        if p_max - p_min < 1e-12:
            mask_rel = np.ones_like(prob, dtype=np.float32)
        else:
            mask_rel = (prob - p_min) / (p_max - p_min)
            mask_rel = mask_rel.astype(np.float32)

        # Step 2: Gaussian smoothing
        try:
            import scipy.ndimage

            mask = scipy.ndimage.gaussian_filter1d(mask_rel, sigma=5)
        except Exception:
            # fallback: simple moving average
            kernel = np.ones(9) / 9.0  # window=9
            mask = np.convolve(mask_rel, kernel, mode="same")

        # ensure in [0,1]
        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    elif masking == "soft-fade-th":
        # p: raw prob
        p = prob.copy()
        th = threshold

        # Step 1: low region soft-fade on (p <= th)
        p_low = p.copy()
        p_low[p > th] = th  # clamp high region to th for normalization

        p_min = p_low.min()
        p_max = th

        if p_max - p_min < 1e-12:
            mask_low = np.ones_like(p_low, dtype=np.float32)
        else:
            mask_low = (p_low - p_min) / (p_max - p_min)
            mask_low = mask_low.astype(np.float32)

        # Step 2: Gaussian smoothing on low region
        try:
            import scipy.ndimage

            mask_low_smooth = scipy.ndimage.gaussian_filter1d(mask_low, sigma=5)
        except Exception:
            kernel = np.ones(9) / 9.0
            mask_low_smooth = np.convolve(mask_low, kernel, mode="same")

        # Step 3: build final mask
        mask = mask_low_smooth
        mask[p > th] = 1.0  # high region full pass

        mask = np.clip(mask, 0.0, 1.0).astype(np.float32)

    else:
        mask = np.ones_like(prob, dtype=np.float32)

    # ============ fallback when silent ============
    if mask.sum() < 1e-6:
        if debug:
            print("[DEBUG] SED mask all-zero → fallback to all-ones")
        mask[:] = 1.0

    # =====================================================================
    #                          DEBUG VISUALIZATION
    # =====================================================================
    if debug and vis_dir is not None:
        import matplotlib.pyplot as plt

        os.makedirs(os.path.join(vis_dir, "sed"), exist_ok=True)
        sed_dir = os.path.join(vis_dir, "sed")

        # -------- save raw probability --------
        plt.figure(figsize=(12, 4))
        plt.plot(prob)
        plt.title(f"SED Probability — class={labels[class_id]}")
        plt.xlabel("Frame index")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.savefig(f"{sed_dir}/01_prob_curve.png")
        plt.close()

        # -------- save mask --------
        plt.figure(figsize=(12, 4))
        plt.plot(mask)
        plt.title(f"SED Mask ({masking})")
        plt.xlabel("Frame index")
        plt.ylabel("Mask value")
        plt.tight_layout()
        plt.savefig(f"{sed_dir}/02_mask_curve.png")
        plt.close()

        # -------- save heatmap --------
        plt.figure(figsize=(10, 4))
        plt.imshow(prob[None, :], aspect="auto", cmap="jet", vmin=0, vmax=1)
        plt.colorbar()
        plt.title(f"Framewise Prob Heatmap — {labels[class_id]}")
        plt.tight_layout()
        plt.savefig(f"{sed_dir}/03_heatmap.png")
        plt.close()

        # -------- onset/offset detection (official style) --------

        def detect_intervals(activity_vec):
            events = []
            active = False
            onset = None
            for i, v in enumerate(activity_vec):
                if v and not active:
                    active = True
                    onset = i
                if not v and active:
                    active = False
                    events.append((onset, i))
            if active:
                events.append((onset, len(activity_vec)))
            return events

        activity = prob > threshold
        intervals = detect_intervals(activity)

        # Save intervals to txt
        with open(f"{sed_dir}/04_intervals.txt", "w") as f:
            f.write(f"SED Intervals for class={labels[class_id]}\n")
            f.write(f"Threshold={threshold}\n\n")
            for a, b in intervals:
                f.write(f"{a*0.01:.2f}s → {b*0.01:.2f}s\n")  # approx time
        print(f"[DEBUG] SED visualization saved in {sed_dir}")

    return prob, mask


# ========== AudioSep ==========
def run_audiosep_inference(wav, sr, text_prompt, device, outputpath=None, debug=False):

    model = build_audiosep(
        config_yaml="./AudioSep/config/audiosep_base.yaml",
        checkpoint_path="./AudioSep/checkpoint/audiosep_base_4M_steps.ckpt",
        device=device,
    )

    mixture = torch.tensor(wav).float()[None, None].to(device)
    cond = model.query_encoder.get_query_embed(
        modality="text", text=[text_prompt], device=device
    )

    with torch.no_grad():
        out = model.ss_model({"mixture": mixture, "condition": cond})["waveform"]

    out = out.squeeze().cpu().numpy()

    if outputpath is not None:
        import soundfile as sf

        sf.write(outputpath, out, sr)
        print(f"[INFO] AudioSep result saved → {outputpath}")

    return out  # 1D numpy array (sr Hz)


# =====================================================
# ========== FlowSep (CWD FIX + RETURN WAVE) ==========
# =====================================================
def run_flowsep_inference(
    text_prompt, audio_path, output_path=None, save_wave=True, debug=False
):
    """
    FlowSep 원본 파이프라인을 그대로 따라가되,
    - return: (waveform_16k, random_start_sample)
    - 필요할 경우 output_path(16k)로 저장
    """

    original_cwd = os.getcwd()
    flowsep_root = os.path.join(original_cwd, "FlowSep")
    os.chdir(flowsep_root)

    sys.path.append(os.path.join(flowsep_root, "src"))

    from latent_diffusion.util import instantiate_from_config
    from utilities.data.dataset import AudioDataset
    import yaml

    config_path = "lass_config/2channel_flow.yaml"
    ckpt_path = "model_logs/pretrained/v2_100k.ckpt"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"[FlowSep] config not found: {config_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[FlowSep] checkpoint not found: {ckpt_path}")

    config_yaml = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    dataset = AudioDataset(config_yaml, split="test", add_ons=[])

    model = instantiate_from_config(config_yaml["model"]).cuda()
    state = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state, strict=True)

    if debug:
        print("[INFO] FlowSep checkpoint loaded:", ckpt_path)

    # === FlowSep 입력 (16k, 10.24초 세그먼트 + random_start) ===
    noise_waveform, random_start = dataset.read_wav_file(
        audio_path
    )  # waveform: (1, T), sr=16000
    noise_waveform = noise_waveform[0]  # (T,)
    mixed_mel, stft = dataset.wav_feature_extraction(noise_waveform.reshape(1, -1))

    batch = {
        "fname": [audio_path],
        "text": [text_prompt],
        "caption": [text_prompt],
        "waveform": torch.rand(1, 1, 163840).cuda(),  # dummy
        "log_mel_spec": torch.rand(1, 1024, 64).cuda(),  # dummy
        "sampling_rate": torch.tensor([16000]).cuda(),
        "label_vector": torch.rand(1, 527).cuda(),  # dummy
        "stft": torch.rand(1, 1024, 512).cuda(),  # dummy
        "mixed_waveform": torch.from_numpy(noise_waveform.reshape(1, 1, -1)),  # (1,1,T)
        "mixed_mel": mixed_mel.reshape(1, mixed_mel.shape[0], mixed_mel.shape[1]),
    }

    # FlowSep generate_sample 호출 (디스크 저장은 save_wave 플래그로 제어)
    waveform = model.generate_sample(
        [batch],
        name="lass_result",  # 내부적으로만 사용, 여기서는 실제 경로로 안 씀
        unconditional_guidance_scale=1.0,
        ddim_steps=20,
        n_gen=1,
        save=save_wave,
        save_mixed=False,
    )

    # waveform: Tensor(B, C, T) 혹은 numpy
    if isinstance(waveform, torch.Tensor):
        wf = waveform.detach().cpu().numpy()
    else:
        wf = np.asarray(waveform)

    # 첫 번째 샘플/채널만 사용
    if wf.ndim == 3:
        wf = wf[0, 0]  # (T,)
    elif wf.ndim == 2:
        wf = wf[0]  # (T,)

    # 원하는 경로에 16k로 저장 (FlowSep 단독/BOTh일 때)
    if save_wave and output_path is not None:
        import soundfile as sf

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, wf, 16000)
        print(f"[INFO] FlowSep result saved → {output_path}")

    os.chdir(original_cwd)

    return wf, int(random_start)  # 16k waveform, start sample (16k 기준)


# =====================================================
# ========== Ensemble A (STFT + magnitude blend) ======
# =====================================================
def ensemble_stft_fusion(
    audiosep_wav_32k, flowsep_wav_16k, random_start_16k, ensemble_rate=0.3, debug=False
):
    """
    Ensemble-A (개선된 버전):
    - AudioSep 32k → 16k 다운샘플
    - FlowSep 구간 RMS 레벨을 AudioSep 구간 RMS 레벨에 맞춰 정규화
    - magnitude blend (alpha)
    """

    # 1) AudioSep 32k → 16k
    audiosep_16k = librosa.resample(audiosep_wav_32k, orig_sr=32000, target_sr=16000)

    flow = np.asarray(flowsep_wav_16k).astype(np.float32)
    L_f = len(flow)

    start = max(int(random_start_16k), 0)
    if debug:
        print(f"[ENSEMBLE-A] normal: start={start}, len={L_f}, alpha={ensemble_rate}")

    # 필요한 경우 AudioSep 확장
    if len(audiosep_16k) < start + L_f:
        pad = start + L_f - len(audiosep_16k)
        audiosep_16k = np.concatenate(
            [audiosep_16k, np.zeros(pad, dtype=audiosep_16k.dtype)]
        )

    seg = audiosep_16k[start : start + L_f].astype(np.float32)

    # ================================
    #  ⭐ 1) RMS 계산 및 FlowSep RMS 정규화
    # ================================
    def rms(x):
        return np.sqrt(np.mean(x**2) + 1e-12)

    rms_audiosep = rms(seg)
    rms_flowsep = rms(flow)

    if debug:
        print(f"[RMS] AudioSep={rms_audiosep:.6f}, FlowSep={rms_flowsep:.6f}")

    # FlowSep 레벨을 AudioSep RMS에 맞춤
    scale = rms_audiosep / (rms_flowsep + 1e-12)
    flow_norm = flow * scale

    if debug:
        print(f"[RMS] scale factor applied: {scale:.6f}")

    # ================================
    #  ⭐ 2) STFT → magnitude blend
    # ================================
    n_fft = 1024
    hop = 256

    S1 = librosa.stft(seg, n_fft=n_fft, hop_length=hop)
    S2 = librosa.stft(flow_norm, n_fft=n_fft, hop_length=hop)

    T = min(S1.shape[1], S2.shape[1])
    S1 = S1[:, :T]
    S2 = S2[:, :T]

    mag1 = np.abs(S1)
    mag2 = np.abs(S2)
    phase1 = np.angle(S1)

    alpha = max(0.0, min(float(ensemble_rate), 1.0))

    # magnitude blend
    M = (1.0 - alpha) * mag1 + alpha * mag2
    S_ens = M * np.exp(1j * phase1)

    # ISTFT
    seg_ens = librosa.istft(S_ens, hop_length=hop, length=seg.shape[0])

    # overlap-add
    audiosep_16k[start : start + len(seg_ens)] = seg_ens.astype(np.float32)

    # 16k → 32k
    out_32k = librosa.resample(audiosep_16k, orig_sr=16000, target_sr=32000)
    out_32k = np.clip(out_32k, -1.0, 1.0)

    return out_32k


# =====================================================
# ========== Ensemble B (저/고주파 분할) ===============
# =====================================================
def ensemble_stft_fusion_freq(
    audiosep_wav_32k, flowsep_wav_16k, random_start_16k, cutoff_hz=4000.0, debug=False
):
    """
    방법2 (ensemble_b):
    - AudioSep 출력(32k)을 16k로 다운샘플
    - FlowSep 영역: random_start_16k ~ random_start_16k + len(flowsep)
    - STFT에서 주파수 축을 기준으로:
        * f <= cutoff_hz  → AudioSep magnitude 사용
        * f >  cutoff_hz  → FlowSep magnitude 사용
      (phase는 AudioSep 사용)
    - 다시 time-domain으로 복원, 전체 신호에 overlap-add
    - 마지막에 32k로 리샘플해서 리턴
    """

    # 1) AudioSep 32k → 16k
    sr = 16000
    audiosep_16k = librosa.resample(audiosep_wav_32k, orig_sr=32000, target_sr=sr)

    flow = np.asarray(flowsep_wav_16k).astype(np.float32)
    L_f = len(flow)

    start = max(int(random_start_16k), 0)
    if debug:
        print(
            f"[ENSEMBLE-B] FlowSep segment start (16k samples): {start}, len={L_f}, cutoff={cutoff_hz} Hz"
        )

    # 2) 필요시 pad
    if len(audiosep_16k) < start + L_f:
        pad = start + L_f - len(audiosep_16k)
        if debug:
            print(f"[ENSEMBLE-B] Padding AudioSep tail by {pad} samples.")
        audiosep_16k = np.concatenate(
            [audiosep_16k, np.zeros(pad, dtype=audiosep_16k.dtype)]
        )

    seg = audiosep_16k[start : start + L_f]

    # 3) STFT
    n_fft = 1024
    hop = 256

    S1 = librosa.stft(seg, n_fft=n_fft, hop_length=hop)  # (F, T)
    S2 = librosa.stft(flow, n_fft=n_fft, hop_length=hop)

    # 시간축 길이 맞추기
    T = min(S1.shape[1], S2.shape[1])
    S1 = S1[:, :T]
    S2 = S2[:, :T]

    mag1 = np.abs(S1)
    mag2 = np.abs(S2)
    phase1 = np.angle(S1)

    F = S1.shape[0]
    # librosa STFT: F = n_fft//2 + 1, freq bins from 0 ~ sr/2
    freqs = np.linspace(0.0, sr / 2.0, F)

    # cutoff 클리핑
    cutoff = float(cutoff_hz)
    if cutoff <= 0.0:
        # 전 대역 FlowSep
        low_mask = np.zeros_like(freqs, dtype=bool)
    elif cutoff >= sr / 2.0:
        # 전 대역 AudioSep
        low_mask = np.ones_like(freqs, dtype=bool)
    else:
        low_mask = freqs <= cutoff

    high_mask = ~low_mask

    if debug:
        print(f"[ENSEMBLE-B] low bins: {low_mask.sum()}, high bins: {high_mask.sum()}")

    # 4) 저주파수: AudioSep, 고주파수: FlowSep
    M = np.zeros_like(mag1)
    M[low_mask, :] = mag1[low_mask, :]
    M[high_mask, :] = mag2[high_mask, :]

    S_ens = M * np.exp(1j * phase1)

    seg_ens = librosa.istft(S_ens, hop_length=hop, length=seg.shape[0])

    # 5) overlap-add
    audiosep_16k[start : start + len(seg_ens)] = seg_ens.astype(audiosep_16k.dtype)

    # 6) 다시 32k로 리샘플
    out_32k = librosa.resample(audiosep_16k, orig_sr=sr, target_sr=32000)
    out_32k = np.clip(out_32k, -1.0, 1.0)

    return out_32k


def ensemble_stft_fusion_c(
    audiosep_wav_32k, flowsep_wav_16k, random_start_16k, cutoff_hz=4000.0, debug=False
):
    """
    Ensemble-C:
      - Low freq: FlowSep magnitude
      - High freq: AudioSep magnitude
      (ensemble_b의 반대 버전)
    """

    sr = 16000
    audiosep_16k = librosa.resample(audiosep_wav_32k, orig_sr=32000, target_sr=16000)

    flow = np.asarray(flowsep_wav_16k).astype(np.float32)
    L_f = len(flow)

    start = max(int(random_start_16k), 0)

    if len(audiosep_16k) < start + L_f:
        pad = start + L_f - len(audiosep_16k)
        audiosep_16k = np.concatenate(
            [audiosep_16k, np.zeros(pad, dtype=audiosep_16k.dtype)]
        )

    seg = audiosep_16k[start : start + L_f]

    # STFT
    n_fft = 1024
    hop = 256
    S1 = librosa.stft(seg, n_fft=n_fft, hop_length=hop)
    S2 = librosa.stft(flow, n_fft=n_fft, hop_length=hop)

    T = min(S1.shape[1], S2.shape[1])
    S1 = S1[:, :T]
    S2 = S2[:, :T]

    mag1 = np.abs(S1)
    mag2 = np.abs(S2)
    phase1 = np.angle(S1)

    F = mag1.shape[0]
    freqs = np.linspace(0, sr / 2, F)

    cutoff = float(cutoff_hz)
    if cutoff <= 0:
        low_mask = np.ones_like(freqs, dtype=bool)
    elif cutoff >= sr / 2:
        low_mask = np.zeros_like(freqs, dtype=bool)
    else:
        low_mask = freqs <= cutoff
    high_mask = ~low_mask

    if debug:
        print(
            f"[ENSEMBLE-C] low_mask(FlowSep): {low_mask.sum()}, high_mask(AudioSep): {high_mask.sum()}"
        )

    # Low freq = FlowSep / High freq = AudioSep
    M = np.zeros_like(mag1)
    M[low_mask, :] = mag2[low_mask, :]
    M[high_mask, :] = mag1[high_mask, :]

    S_ens = M * np.exp(1j * phase1)
    seg_ens = librosa.istft(S_ens, hop_length=hop, length=len(seg))

    audiosep_16k[start : start + len(seg_ens)] = seg_ens
    out_32k = librosa.resample(audiosep_16k, orig_sr=sr, target_sr=32000)
    out_32k = np.clip(out_32k, -1, 1)
    return out_32k


def ensemble_stft_fusion_d(
    audiosep_wav_32k, flowsep_wav_16k, random_start_16k, ensemble_rate=0.3, debug=False
):
    """
    Ensemble-D:
      - A + C 결합 버전
      - 저주파수: AudioSep 비중 ensemble_rate 쪽으로 치우침
      - 고주파수: FlowSep 비중이 점차 1.0으로 증가
      즉,
         weight_flowsep(f) = (1 - ensemble_rate) + (ensemble_rate) * (f_norm)
         weight_audiosep(f)  = 1 - weight_flowsep(f)
      (f_norm = freq bin normalized 0~1)
    """

    sr = 16000
    audiosep_16k = librosa.resample(audiosep_wav_32k, orig_sr=32000, target_sr=sr)

    flow = np.asarray(flowsep_wav_16k).astype(np.float32)
    L_f = len(flow)
    start = max(int(random_start_16k), 0)

    if len(audiosep_16k) < start + L_f:
        pad = start + L_f - len(audiosep_16k)
        audiosep_16k = np.concatenate([audiosep_16k, np.zeros(pad, dtype=np.float32)])

    seg = audiosep_16k[start : start + L_f]

    # STFT
    n_fft = 1024
    hop = 256
    S1 = librosa.stft(seg, n_fft=n_fft, hop_length=hop)
    S2 = librosa.stft(flow, n_fft=n_fft, hop_length=hop)

    T = min(S1.shape[1], S2.shape[1])
    S1 = S1[:, :T]
    S2 = S2[:, :T]

    mag1 = np.abs(S1)
    mag2 = np.abs(S2)
    phase1 = np.angle(S1)

    F = mag1.shape[0]
    freqs = np.linspace(0, sr / 2, F)
    f_norm = (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-12)

    er = float(ensemble_rate)

    weight_audio = (1 - er) + er * (1.0 - f_norm)
    weight_flow = 1.0 - weight_audio

    if debug:
        print(
            f"[ENSEMBLE-D-REV] audio_weight[0]={weight_audio[0]:.3f}, audio_weight[-1]={weight_audio[-1]:.3f}"
        )

    # freq-wise magnitude fuse
    M = mag1 * weight_audio[:, None] + mag2 * weight_flow[:, None]
    S_ens = M * np.exp(1j * phase1)

    seg_ens = librosa.istft(S_ens, hop_length=hop, length=len(seg))
    audiosep_16k[start : start + len(seg_ens)] = seg_ens

    out_32k = librosa.resample(audiosep_16k, orig_sr=sr, target_sr=32000)
    return np.clip(out_32k, -1, 1)


def ensemble_stft_fusion_e(
    audiosep_wav_32k, flowsep_wav_16k, random_start_16k, ensemble_rate=0.3, debug=False
):
    """
    Ensemble-E:
      - A + C 결합 버전
      - 저주파수: FlowSep 비중 ensemble_rate 쪽으로 치우침
      - 고주파수: AudioSep 비중이 점차 1.0으로 증가
      즉,
         weight_audiosep(f) = (1 - ensemble_rate) + (ensemble_rate) * (f_norm)
         weight_flowsep(f)  = 1 - weight_audiosep(f)
      (f_norm = freq bin normalized 0~1)
    """

    sr = 16000
    audiosep_16k = librosa.resample(audiosep_wav_32k, orig_sr=32000, target_sr=sr)

    flow = np.asarray(flowsep_wav_16k).astype(np.float32)
    L_f = len(flow)
    start = max(int(random_start_16k), 0)

    if len(audiosep_16k) < start + L_f:
        pad = start + L_f - len(audiosep_16k)
        audiosep_16k = np.concatenate([audiosep_16k, np.zeros(pad, dtype=np.float32)])

    seg = audiosep_16k[start : start + L_f]

    # STFT
    n_fft = 1024
    hop = 256
    S1 = librosa.stft(seg, n_fft=n_fft, hop_length=hop)
    S2 = librosa.stft(flow, n_fft=n_fft, hop_length=hop)

    T = min(S1.shape[1], S2.shape[1])
    S1 = S1[:, :T]
    S2 = S2[:, :T]

    mag1 = np.abs(S1)
    mag2 = np.abs(S2)
    phase1 = np.angle(S1)

    F = mag1.shape[0]
    freqs = np.linspace(0, sr / 2, F)
    f_norm = (freqs - freqs.min()) / (freqs.max() - freqs.min() + 1e-12)

    er = float(ensemble_rate)

    # weight mapping
    # high freq → AudioSep→1.0
    weight_audio = (1 - er) + er * f_norm
    weight_flow = 1.0 - weight_audio

    M = mag1 * weight_audio[:, None] + mag2 * weight_flow[:, None]
    S_ens = M * np.exp(1j * phase1)

    seg_ens = librosa.istft(S_ens, hop_length=hop, length=len(seg))
    audiosep_16k[start : start + len(seg_ens)] = seg_ens

    out_32k = librosa.resample(audiosep_16k, orig_sr=sr, target_sr=32000)
    return np.clip(out_32k, -1, 1)


# =====================================================
# ========== Unified pipeline ==========================
# =====================================================
def run_unified(args):

    import soundfile as sf

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wav = load_audio(args.audio, sr=32000)

    # ===== NEW: visualization output root =====
    vis_root = args.vis_dir or "output/visualize"
    os.makedirs(vis_root, exist_ok=True)

    # Save input audio & spectrogram
    save_waveform(f"{vis_root}/01_input_audio.wav", wav, 32000)
    save_spectrogram(f"{vis_root}/01_input_audio.png", wav, 32000)

    if args.gt_audio is not None:
        gt, _ = librosa.load(args.gt_audio, sr=32000, mono=True)
        save_waveform(f"{vis_root}/00_gt.wav", gt, 32000)
        save_spectrogram(f"{vis_root}/00_gt.png", gt, 32000)

    # ====== SED + masking ======
    _, mask = run_panns_sed_mask(
        wav,
        device,
        args.text,
        threshold=args.threshold,
        masking=args.masking,
        debug=args.debug,
        vis_dir=vis_root,  # ← 추가된 부분
    )

    mask_interp = interp_mask(mask, len(wav))
    wav_masked = wav * mask_interp

    if args.masking in ["hard", "soft", "soft-rel", "soft-fade", "soft-fade-th"]:
        save_waveform(f"{vis_root}/02_sed_masked.wav", wav_masked, 32000)
        save_spectrogram(f"{vis_root}/02_sed_masked.png", wav_masked, 32000)

    # ========= AudioSep ONLY =========
    if args.base_model == "audiosep":
        if args.output_audiosep is None:
            raise ValueError("AudioSep requires --output_audiosep")

        audiosep_out = run_audiosep_inference(
            wav_masked, 32000, args.text, device, args.output_audiosep, debug=args.debug
        )

        # NEW: Visualization
        save_waveform(f"{vis_root}/03_audiosep.wav", audiosep_out, 32000)
        save_spectrogram(f"{vis_root}/03_audiosep.png", audiosep_out, 32000)
        return

    # ========= FlowSep ONLY =========
    if args.base_model == "flowsep":
        if args.output_flowsep is None:
            raise ValueError("FlowSep requires --output_flowsep")

        wav16 = librosa.resample(wav_masked, orig_sr=32000, target_sr=16000)
        temp_input = os.path.abspath("_temp_flow.wav")
        sf.write(temp_input, wav16, 16000)

        flow_out, _ = run_flowsep_inference(
            text_prompt=args.text,
            audio_path=temp_input,
            output_path=args.output_flowsep,
            save_wave=True,
            debug=args.debug,
        )

        # NEW: Visualization
        save_waveform(f"{vis_root}/04_flowsep.wav", flow_out, 16000)
        save_spectrogram(f"{vis_root}/04_flowsep.png", flow_out, 16000)
        return

    # ========= BOTH (AudioSep + FlowSep) =========
    if args.base_model == "both":
        if args.output_audiosep is None or args.output_flowsep is None:
            raise ValueError("Both requires --output_audiosep and --output_flowsep")

        # AudioSep
        audiosep_out = run_audiosep_inference(
            wav_masked, 32000, args.text, device, args.output_audiosep, debug=args.debug
        )
        save_waveform(f"{vis_root}/03_audiosep.wav", audiosep_out, 32000)
        save_spectrogram(f"{vis_root}/03_audiosep.png", audiosep_out, 32000)

        # FlowSep
        wav16 = librosa.resample(wav_masked, orig_sr=32000, target_sr=16000)
        temp_input = os.path.abspath("_temp_flow.wav")
        sf.write(temp_input, wav16, 16000)

        flow_out, _ = run_flowsep_inference(
            args.text, temp_input, args.output_flowsep, save_wave=True, debug=args.debug
        )
        save_waveform(f"{vis_root}/04_flowsep.wav", flow_out, 16000)
        save_spectrogram(f"{vis_root}/04_flowsep.png", flow_out, 16000)

        return

    # ========= ENSEMBLE_A (AudioSep + FlowSep magnitude blend) =========
    if args.base_model == "ensemble_a":
        if args.output_audiosep is None:
            raise ValueError(
                "ensemble_a requires --output_audiosep (final ensemble output)"
            )

        # 1) AudioSep (32k) - 파일 저장 없이 메모리에서만 사용
        audiosep_out = run_audiosep_inference(
            wav_masked, 32000, args.text, device, outputpath=None, debug=args.debug
        )

        # NEW: save AudioSep visual
        save_waveform(f"{vis_root}/03_audiosep.wav", audiosep_out, 32000)
        save_spectrogram(f"{vis_root}/03_audiosep.png", audiosep_out, 32000)

        # 2) FlowSep (16k, random_start)
        wav16 = librosa.resample(wav_masked, orig_sr=32000, target_sr=16000)
        temp_input = os.path.abspath("_temp_flow.wav")
        sf.write(temp_input, wav16, 16000)

        flow_wav_16k, random_start = run_flowsep_inference(
            text_prompt=args.text,
            audio_path=temp_input,
            output_path=None,  # ensemble에서는 FlowSep 단독 결과 파일 저장 X
            save_wave=False,
            debug=args.debug,
        )

        # 3) STFT 기반 정렬 + 앙상블
        ensemble_rate = float(args.ensemble_rate)
        ens_32k = ensemble_stft_fusion(
            audiosep_wav_32k=audiosep_out,
            flowsep_wav_16k=flow_wav_16k,
            random_start_16k=random_start,
            ensemble_rate=ensemble_rate,
            debug=args.debug,
        )

        # 4) 최종 결과 저장 (32k)
        os.makedirs(os.path.dirname(args.output_audiosep) or ".", exist_ok=True)
        sf.write(args.output_audiosep, ens_32k, 32000)
        # NEW: Save ensemble result
        save_waveform(f"{vis_root}/05_ensemble.wav", ens_32k, 32000)
        save_spectrogram(f"{vis_root}/05_ensemble.png", ens_32k, 32000)
        print(f"[INFO] Ensemble-A result saved → {args.output_audiosep}")
        return

    # ========= ENSEMBLE_B (저/고주파 분할) =========
    if args.base_model == "ensemble_b":
        if args.output_audiosep is None:
            raise ValueError(
                "ensemble_b requires --output_audiosep (final ensemble output)"
            )

        # 1) AudioSep (32k)
        audiosep_out = run_audiosep_inference(
            wav_masked, 32000, args.text, device, outputpath=None, debug=args.debug
        )

        # 2) FlowSep (16k, random_start)
        wav16 = librosa.resample(wav_masked, orig_sr=32000, target_sr=16000)
        temp_input = os.path.abspath("_temp_flow.wav")
        sf.write(temp_input, wav16, 16000)

        flow_wav_16k, random_start = run_flowsep_inference(
            text_prompt=args.text,
            audio_path=temp_input,
            output_path=None,
            save_wave=False,
            debug=args.debug,
        )
        # NEW: Save FlowSep results
        save_waveform(f"{vis_root}/04_flowsep.wav", flow_wav_16k, 16000)
        save_spectrogram(f"{vis_root}/04_flowsep.png", flow_wav_16k, 16000)

        # 3) 주파수 분할 기반 앙상블
        cutoff = float(args.ensemble_freq)
        ens_32k = ensemble_stft_fusion_freq(
            audiosep_wav_32k=audiosep_out,
            flowsep_wav_16k=flow_wav_16k,
            random_start_16k=random_start,
            cutoff_hz=cutoff,
            debug=args.debug,
        )

        # 4) 최종 결과 저장 (32k)
        os.makedirs(os.path.dirname(args.output_audiosep) or ".", exist_ok=True)
        sf.write(args.output_audiosep, ens_32k, 32000)
        # NEW: Save ensemble result
        save_waveform(f"{vis_root}/05_ensemble.wav", ens_32k, 32000)
        save_spectrogram(f"{vis_root}/05_ensemble.png", ens_32k, 32000)
        print(f"[INFO] Ensemble-B result saved → {args.output_audiosep}")
        return

        # ========= ENSEMBLE_C =========
    if args.base_model == "ensemble_c":
        if args.output_audiosep is None:
            raise ValueError("ensemble_c requires --output_audiosep")

        audiosep_out = run_audiosep_inference(
            wav_masked, 32000, args.text, device, outputpath=None, debug=args.debug
        )

        wav16 = librosa.resample(wav_masked, orig_sr=32000, target_sr=16000)
        temp_input = os.path.abspath("_temp_flow.wav")
        sf.write(temp_input, wav16, 16000)

        flow_wav_16k, random_start = run_flowsep_inference(
            args.text, temp_input, None, save_wave=False, debug=args.debug
        )

        ens_32k = ensemble_stft_fusion_c(
            audiosep_out,
            flow_wav_16k,
            random_start,
            cutoff_hz=float(args.ensemble_freq),
            debug=args.debug,
        )

        sf.write(args.output_audiosep, ens_32k, 32000)
        # NEW: Save ensemble result
        save_waveform(f"{vis_root}/05_ensemble.wav", ens_32k, 32000)
        save_spectrogram(f"{vis_root}/05_ensemble.png", ens_32k, 32000)
        print("[INFO] Ensemble-C saved")
        return

    # ========= ENSEMBLE_D =========
    if args.base_model == "ensemble_d":
        if args.output_audiosep is None:
            raise ValueError("ensemble_d requires --output_audiosep")

        audiosep_out = run_audiosep_inference(
            wav_masked, 32000, args.text, device, outputpath=None, debug=args.debug
        )

        wav16 = librosa.resample(wav_masked, orig_sr=32000, target_sr=16000)
        temp_input = os.path.abspath("_temp_flow.wav")
        sf.write(temp_input, wav16, 16000)

        flow_wav_16k, random_start = run_flowsep_inference(
            args.text, temp_input, None, save_wave=False, debug=args.debug
        )

        ens_32k = ensemble_stft_fusion_d(
            audiosep_out,
            flow_wav_16k,
            random_start,
            ensemble_rate=float(args.ensemble_rate),
            debug=args.debug,
        )

        sf.write(args.output_audiosep, ens_32k, 32000)
        # NEW: Save ensemble result
        save_waveform(f"{vis_root}/05_ensemble.wav", ens_32k, 32000)
        save_spectrogram(f"{vis_root}/05_ensemble.png", ens_32k, 32000)
        print("[INFO] Ensemble-D saved")
        return

    # ========= ENSEMBLE_E =========
    if args.base_model == "ensemble_e":
        if args.output_audiosep is None:
            raise ValueError("ensemble_e requires --output_audiosep")

        audiosep_out = run_audiosep_inference(
            wav_masked, 32000, args.text, device, outputpath=None, debug=args.debug
        )

        wav16 = librosa.resample(wav_masked, orig_sr=32000, target_sr=16000)
        temp_input = os.path.abspath("_temp_flow.wav")
        sf.write(temp_input, wav16, 16000)

        flow_wav_16k, random_start = run_flowsep_inference(
            args.text, temp_input, None, save_wave=False, debug=args.debug
        )

        ens_32k = ensemble_stft_fusion_e(
            audiosep_out,
            flow_wav_16k,
            random_start,
            ensemble_rate=float(args.ensemble_rate),
            debug=args.debug,
        )

        sf.write(args.output_audiosep, ens_32k, 32000)
        # NEW: Save ensemble result
        save_waveform(f"{vis_root}/05_ensemble.wav", ens_32k, 32000)
        save_spectrogram(f"{vis_root}/05_ensemble.png", ens_32k, 32000)
        print("[INFO] Ensemble-E saved")
        return

    raise ValueError(f"Unknown base_model: {args.base_model}")


# =====================================================
# ========== Main ======================================
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument(
        "--masking",
        choices=["none", "hard", "soft", "soft-rel", "soft-fade", "soft-fade-th"],
        default="hard",
    )
    parser.add_argument("--threshold", type=float, default=0.3)

    parser.add_argument(
        "--base_model",
        choices=[
            "audiosep",
            "flowsep",
            "both",
            "ensemble_a",
            "ensemble_b",
            "ensemble_c",
            "ensemble_d",
            "ensemble_e",
        ],
        default="audiosep",
    )

    parser.add_argument("--output_audiosep", default=None)
    parser.add_argument("--output_flowsep", default=None)

    # ⭐ ensemble_a용 가중치 (FlowSep 비율 alpha, 0.0~1.0)
    parser.add_argument(
        "--ensemble_rate",
        type=float,
        default=0.3,
        help="Ensemble-A weight for FlowSep (0.0~1.0). 0이면 AudioSep만, 1이면 FlowSep magnitude만 사용.",
    )

    # ⭐ ensemble_b용 주파수 분기점 (Hz, 16k 기준, 0~8000)
    parser.add_argument(
        "--ensemble_freq",
        type=float,
        default=4000.0,
        help="Ensemble-B cutoff frequency in Hz (low: AudioSep, high: FlowSep) on 16k STFT.",
    )

    parser.add_argument("--debug", action="store_true")

    # NEW
    parser.add_argument("--gt_audio", default=None, help="Ground-truth audio path")
    parser.add_argument(
        "--vis_dir",
        default="output/visualize",
        help="Directory to save wave & spectrogram images",
    )

    args = parser.parse_args()

    run_unified(args)
