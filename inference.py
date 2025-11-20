# # import os
# # import sys
# # import torch
# # import numpy as np
# # import librosa
# # import argparse

# # # === AudioSep imports ===
# # sys.path.append("./AudioSep")
# # from pipeline import build_audiosep
# # from models.clap_encoder import CLAP_Encoder

# # # === PANNs imports ===
# # sys.path.append("./audioset_tagging_cnn/pytorch")
# # from audioset_tagging_cnn.pytorch.models import Cnn14_DecisionLevelAtt
# # from pytorch_utils import move_data_to_device

# # # === util ===
# # def load_audio(path, sr=32000):
# #     wav, _ = librosa.load(path, sr=sr, mono=True)
# #     return wav


# # # ----------------------------------------------------------------------
# # # (1) PANNs SED INFERENCE
# # # ----------------------------------------------------------------------
# # def run_panns_sed(audio, device):
# #     """
# #     audio: numpy waveform (32kHz)
# #     returns: temporal_mask (T_audioSep)
# #     """

# #     # === Load PANNs model ===
# #     model_path = "./audioset_tagging_cnn/pytorch/Cnn14_DecisionLevelAtt_mAP=0.425.pth"
# #     model = Cnn14_DecisionLevelAtt(
# #         sample_rate=32000,
# #         window_size=1024,
# #         hop_size=320,
# #         mel_bins=64,
# #         fmin=50,
# #         fmax=14000,
# #         classes_num=527
# #     )

# #     checkpoint = torch.load(model_path, map_location=device)
# #     model.load_state_dict(checkpoint["model"])
# #     model.to(device).eval()

# #     # === Prepare audio ===
# #     wav_tensor = torch.tensor(audio).float()[None, :].to(device)

# #     with torch.no_grad():
# #         out = model(wav_tensor, None)

# #     framewise = out["framewise_output"][0].cpu().numpy()   # (T_panns, 527)

# #     # ------------------------------------------------------------------
# #     # DEFINE SPEECH-RELATED AudioSet classes
# #     # ------------------------------------------------------------------
# #     # Minimal (you can extend later)
# #     speech_like = [
# #         0,   # Speech
# #         1,   # Male speech
# #         2,   # Female speech
# #         3,   # Child speech
# #         4,   # Conversation
# #         5,   # Narration
# #         6,   # Singing
# #     ]

# #     # Compute speech activity per frame
# #     speech_prob = framewise[:, speech_like].max(axis=1)

# #     # Threshold
# #     th = 0.4
# #     speech_activity = (speech_prob > th).astype(float)   # shape (T_panns,)

# #     return speech_activity


# # # ----------------------------------------------------------------------
# # # (2) INTERPOLATE TO AUDIOSEP TIME AXIS
# # # ----------------------------------------------------------------------
# # def interp_temporal_mask(panns_mask, target_len):
# #     """
# #     panns_mask: (T_panns,)
# #     target_len: AudioSep waveform length
# #     """
# #     pT = len(panns_mask)
# #     x_src = np.linspace(0, 1, pT)
# #     x_tgt = np.linspace(0, 1, target_len)
# #     return np.interp(x_tgt, x_src, panns_mask)


# # # ----------------------------------------------------------------------
# # # (3) RUN AUDIOSEP WITH TEMPORAL MASK
# # # ----------------------------------------------------------------------
# # def run_audiosep(audio_path, text_prompt, output_path, device):

# #     # --- Load waveform ---
# #     wav = load_audio(audio_path, sr=32000)

# #     # --- SED mask ---
# #     sed_mask = run_panns_sed(wav, device)

# #     # Build AudioSep
# #     model = build_audiosep(
# #         config_yaml="./AudioSep/config/audiosep_base.yaml",
# #         checkpoint_path="./AudioSep/checkpoint/audiosep_base_4M_steps.ckpt",
# #         device=device
# #     )

# #     # ---- Prepare input dict ----
# #     # AudioSep expects mixture shape: (B,1,T)
# #     mixture_tensor = torch.tensor(wav).float()[None, None, :].to(device)

# #     # Text to CLAP embedding
# #     text = [text_prompt]
# #     conditions = model.query_encoder.get_query_embed(
# #         modality="text",
# #         text=text,
# #         device=device
# #     )

# #     input_dict = {
# #         "mixture": mixture_tensor,
# #         "condition": conditions,
# #     }

# #     # Attach temporal mask
# #     # (interpolated to waveform length)
# #     interp_mask = interp_temporal_mask(sed_mask, wav.shape[0])
# #     input_dict["temporal_mask"] = interp_mask

# #     print(f"\n[INFO] Using SED mask with {np.sum(interp_mask>0)} active frames")

# #     # --- Run AudioSep ---
# #     with torch.no_grad():
# #         out = model.ss_model(input_dict)["waveform"]   # (B,1,T)
# #         out = out.squeeze().cpu().numpy()

# #     # --- Save ---
# #     import soundfile as sf
# #     sf.write(output_path, out, 32000)
# #     print(f"[INFO] Saved separated audio to {output_path}")


# # # ----------------------------------------------------------------------
# # # MAIN
# # # ----------------------------------------------------------------------
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--audio", type=str, required=True)
# #     parser.add_argument("--text", type=str, required=True)
# #     parser.add_argument("--output", type=str, default="output.wav")
# #     args = parser.parse_args()

# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #     run_audiosep(
# #         audio_path=args.audio,
# #         text_prompt=args.text,
# #         output_path=args.output,
# #         device=device
# #     )


# import os
# import sys
# import torch
# import numpy as np
# import librosa
# import argparse
# import csv

# # === AudioSep imports ===
# sys.path.append("./AudioSep")
# from pipeline import build_audiosep

# # === PANNs imports ===
# sys.path.append("./audioset_tagging_cnn/pytorch")
# from audioset_tagging_cnn.pytorch.models import Cnn14_DecisionLevelAtt
# from pytorch_utils import move_data_to_device


# # === util ===
# def load_audio(path, sr=32000):
#     wav, _ = librosa.load(path, sr=sr, mono=True)
#     return wav


# def load_audioset_labels():
#     """
#     PANNs repo의 metadata/class_labels_indices.csv에서 527개 클래스 이름을 읽어옴.
#     없으면 None 리턴.
#     """
#     meta_path = os.path.join("audioset_tagging_cnn", "metadata", "class_labels_indices.csv")
#     if not os.path.exists(meta_path):
#         print(f"[WARN] metadata file not found: {meta_path}")
#         return None

#     labels = []
#     with open(meta_path, "r", newline="") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             labels.append(row["display_name"])
#     if len(labels) != 527:
#         print(f"[WARN] Expected 527 labels, got {len(labels)}")
#     return labels


# def choose_class_group(labels, text_prompt, debug=False):
#     """
#     text_prompt 내용을 보고 어떤 class 그룹을 SED 마스크에 사용할지 결정.
#     - speech_group: speech / conversation / narration / voice 등 포함
#     - music_group: music / instrument / singing / vocal / choir / lullaby 등 포함
#     """
#     if labels is None:
#         # fallback: 아무 정보 없으면 speech_group index 대충 하드코딩 (최소 안전장치)
#         speech_like = [0, 1, 2, 3, 4, 5, 6]
#         music_like = speech_like  # 어쩔 수 없이 동일 사용
#         mode = "speech"
#         return mode, speech_like

#     text = (text_prompt or "").lower()

#     # 간단한 heuristic
#     music_keywords = ["lullaby", "music", "song", "sing", "vocal", "choir",
#                       "piano", "guitar", "violin", "instrument", "drum"]
#     if any(kw in text for kw in music_keywords):
#         mode = "music"
#     else:
#         mode = "speech"

#     speech_like_ids = []
#     music_like_ids = []

#     for i, name in enumerate(labels):
#         name_l = name.lower()
#         # speech-like
#         if any(kw in name_l for kw in ["speech", "conversation", "narration",
#                                        "narrator", "talking", "talk", "chant",
#                                        "chanting", "debate", "storytelling"]):
#             speech_like_ids.append(i)
#         # music-like
#         if any(kw in name_l for kw in ["music", "musical", "instrument",
#                                        "singing", "vocal", "choir",
#                                        "humming", "lullaby", "orchestra",
#                                        "band"]):
#             music_like_ids.append(i)

#     if debug:
#         print(f"[DEBUG] choose_class_group: mode={mode}")
#         print(f"[DEBUG] speech_like_ids: {len(speech_like_ids)} classes")
#         print(f"[DEBUG] music_like_ids:  {len(music_like_ids)} classes")

#     if mode == "music" and len(music_like_ids) > 0:
#         return mode, music_like_ids
#     elif mode == "speech" and len(speech_like_ids) > 0:
#         return mode, speech_like_ids
#     else:
#         # fallback: labels 기반 실패 시 전체 중 top classes 활용
#         print("[WARN] No matching classes found for mode, fallback to all classes.")
#         all_ids = list(range(len(labels)))
#         return mode, all_ids


# # ----------------------------------------------------------------------
# # (1) PANNs SED INFERENCE
# # ----------------------------------------------------------------------
# def run_panns_sed(audio, device, text_prompt, debug=False):
#     """
#     audio: numpy waveform (32kHz)
#     returns: temporal_mask (T_panns,)
#     """

#     # === Load PANNs model ===
#     model_path = "./audioset_tagging_cnn/pytorch/Cnn14_DecisionLevelAtt_mAP=0.425.pth"
#     if not os.path.exists(model_path):
#         print(f"[ERROR] PANNs checkpoint not found at {model_path}")
#         raise FileNotFoundError(model_path)

#     model = Cnn14_DecisionLevelAtt(
#         sample_rate=32000,
#         window_size=1024,
#         hop_size=320,
#         mel_bins=64,
#         fmin=50,
#         fmax=14000,
#         classes_num=527,
#     )

#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint["model"])
#     model.to(device).eval()

#     # === Load label names ===
#     labels = load_audioset_labels()

#     # === Prepare audio ===
#     wav_tensor = torch.tensor(audio).float()[None, :].to(device)

#     with torch.no_grad():
#         out = model(wav_tensor, None)

#     framewise = out["framewise_output"][0]  # (T_panns, 527) as torch.Tensor
#     framewise_np = framewise.cpu().numpy()

#     # ---- clipwise 출력도 만들어서 top-10 디버그 ----
#     clipwise = torch.sigmoid(out["clipwise_output"][0]).cpu().numpy()  # (527,)

#     if debug:
#         print("\n=== [DEBUG] PANNs Clipwise Top-10 ===")
#         top_idx = np.argsort(clipwise)[-10:][::-1]
#         for idx in top_idx:
#             if labels is not None and idx < len(labels):
#                 name = labels[idx]
#             else:
#                 name = f"Class{idx}"
#             print(f"{idx:3d} | {name:30s} | {clipwise[idx]:.4f}")
#         print("======================================\n")

#         print(f"[DEBUG] Framewise shape: {framewise_np.shape}")
#         fw_max = framewise_np.max(axis=1)
#         print(f"[DEBUG] Framewise max over classes: min={fw_max.min():.4f}, max={fw_max.max():.4f}")

#     # ------------------------------------------------------------------
#     # 선택된 class group (speech / music)을 기반으로 framewise SED mask 생성
#     # ------------------------------------------------------------------
#     mode, class_group = choose_class_group(labels, text_prompt, debug=debug)

#     if len(class_group) == 0:
#         print("[WARN] class_group is empty, fallback to all classes.")
#         class_group = list(range(framewise_np.shape[1]))

#     group_probs = torch.sigmoid(framewise[:, class_group]).max(dim=1)[0].cpu().numpy()
#     # threshold 조정: speech vs music 약간 다르게 써도 됨
#     if mode == "music":
#         th = 0.3
#     else:
#         th = 0.4

#     mask = (group_probs > th).astype(float)

#     if debug:
#         print(f"[DEBUG] SED mode: {mode}")
#         print(f"[DEBUG] Threshold: {th}")
#         print(f"[DEBUG] Mask shape: {mask.shape}, active_frames={mask.sum()} / {len(mask)}")
#         if mask.sum() > 0:
#             # active frame들의 시간(초) 일부만 출력
#             hop = 320 / 32000.0  # 10 ms
#             active_idx = np.where(mask > 0)[0]
#             times = active_idx * hop
#             print("[DEBUG] First 20 active frame times (sec):")
#             print(times[:20])
#         else:
#             print("[DEBUG] No active frames detected.")

#     # mask가 전부 0이면, 디버깅을 위해 temporal gating을 꺼버림
#     if mask.sum() == 0:
#         print("[WARN] SED mask is all zeros. Disabling temporal gating for this run.")
#         mask = np.ones_like(mask)

#     return mask


# # ----------------------------------------------------------------------
# # (2) INTERPOLATE TO AUDIOSEP TIME AXIS
# # ----------------------------------------------------------------------
# def interp_temporal_mask(panns_mask, target_len, debug=False):
#     """
#     panns_mask: (T_panns,)
#     target_len: AudioSep waveform length
#     """
#     pT = len(panns_mask)
#     x_src = np.linspace(0, 1, pT)
#     x_tgt = np.linspace(0, 1, target_len)
#     out = np.interp(x_tgt, x_src, panns_mask)

#     if debug:
#         print(f"[DEBUG] interp_temporal_mask: from {pT} -> {target_len}")
#         print(f"[DEBUG] interp_mask stats: min={out.min():.4f}, max={out.max():.4f}, mean={out.mean():.4f}")
#     return out


# # ----------------------------------------------------------------------
# # (3) RUN AUDIOSEP WITH TEMPORAL MASK
# # ----------------------------------------------------------------------
# def run_audiosep(audio_path, text_prompt, output_path, device, debug=False):

#     # --- Load waveform ---
#     wav = load_audio(audio_path, sr=32000)
#     if debug:
#         print(f"[DEBUG] Loaded audio: {audio_path}")
#         print(f"[DEBUG] Waveform shape: {wav.shape}, sr=32000, duration={len(wav)/32000.0:.2f}s")

#     # --- SED mask ---
#     sed_mask = run_panns_sed(wav, device, text_prompt, debug=debug)

#     # Build AudioSep
#     model = build_audiosep(
#         config_yaml="./AudioSep/config/audiosep_base.yaml",
#         checkpoint_path="./AudioSep/checkpoint/audiosep_base_4M_steps.ckpt",
#         device=device
#     )

#     # ---- Prepare input dict ----
#     # AudioSep expects mixture shape: (B,1,T)
#     mixture_tensor = torch.tensor(wav).float()[None, None, :].to(device)

#     # Text to CLAP embedding
#     text = [text_prompt]
#     conditions = model.query_encoder.get_query_embed(
#         modality="text",
#         text=text,
#         device=device
#     )

#     input_dict = {
#         "mixture": mixture_tensor,
#         "condition": conditions,
#     }

#     # Attach temporal mask
#     # (interpolated to waveform length)
#     interp_mask = interp_temporal_mask(sed_mask, wav.shape[0], debug=debug)
#     input_dict["temporal_mask"] = interp_mask

#     print(f"\n[INFO] Using SED mask with ~{int((interp_mask > 0.5).sum())} active samples "
#           f"({(interp_mask>0.5).sum()/len(interp_mask)*100:.2f}%)")

#     # --- Run AudioSep ---
#     with torch.no_grad():
#         out = model.ss_model(input_dict)["waveform"]   # (B,1,T) or (B,C,T)
#         if debug:
#             print(f"[DEBUG] AudioSep output waveform shape: {out.shape}")
#         out = out.squeeze().cpu().numpy()

#     # --- Save ---
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     import soundfile as sf
#     sf.write(output_path, out, 32000)
#     print(f"[INFO] Saved separated audio to {output_path}")


# # ----------------------------------------------------------------------
# # MAIN
# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--audio", type=str, required=True)
#     parser.add_argument("--text", type=str, required=True)
#     parser.add_argument("--output", type=str, default="output.wav")
#     parser.add_argument("--debug", action="store_true", help="Print detailed SED / mask debug info")
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     run_audiosep(
#         audio_path=args.audio,
#         text_prompt=args.text,
#         output_path=args.output,
#         device=device,
#         debug=args.debug
#     )




import os
import sys
import torch
import numpy as np
import librosa
import argparse
import csv

# =====================================================
# ==========   AudioSep Imports   ======================
# =====================================================
sys.path.append("./AudioSep")
from pipeline import build_audiosep

# =====================================================
# ==========   PANNs Imports   =========================
# =====================================================
sys.path.append("./audioset_tagging_cnn/pytorch")
from audioset_tagging_cnn.pytorch.models import Cnn14_DecisionLevelAtt


# =====================================================
# ========== Util ======================================
# =====================================================
def load_audio(path, sr=32000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def load_audioset_labels():
    """
    Loads 527 AudioSet class names from metadata.
    """
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
# ========== Class selection via prompt ===============
# =====================================================
def select_target_class(labels, text_prompt):
    """
    Return the class index corresponding to the prompt.
    Example: prompt = "Lullaby" → return index of "Lullaby".
    """

    if labels is None:
        raise ValueError("AudioSet labels not loaded.")

    t = text_prompt.lower()

    # Direct match
    for i, name in enumerate(labels):
        if name.lower() == t:
            return i

    # Partial match ("lullaby music", "lullaby song")
    for i, name in enumerate(labels):
        if t in name.lower():
            return i

    # Last fallback
    print(f"[WARN] Could not find class matching '{text_prompt}', using 'Music'.")
    return labels.index("Music")


# =====================================================
# ========== Onset–Offset Extraction ===================
# =====================================================
def extract_onset_offset(prob, threshold, frame_hop=0.01, min_frames=3):
    """
    prob: (T,) probability curve for a single class
    threshold: probability threshold
    returns: list of (start_sec, end_sec)
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


# =====================================================
# ========== PANNs SED ================================
# =====================================================
def run_panns_sed(audio, device, text_prompt, threshold, debug=False):
    """
    Returns:
        target_segments : [(onset_sec, offset_sec), ... ] for the target class
        temporal_mask   : (T_panns,) binary mask
    """

    # 1. Load model
    ckpt = "./audioset_tagging_cnn/pytorch/Cnn14_DecisionLevelAtt_mAP=0.425.pth"
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"PANNs checkpoint not found: {ckpt}")

    model = Cnn14_DecisionLevelAtt(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model = model.to(device).eval()

    # 2. Load labels
    labels = load_audioset_labels()

    # 3. Select target class (e.g., "Lullaby")
    class_id = select_target_class(labels, text_prompt)
    target_name = labels[class_id]
    if debug:
        print(f"[DEBUG] Target class: {target_name} (id={class_id})")

    # 4. Framewise inference
    wav_tensor = torch.tensor(audio).float()[None].to(device)
    with torch.no_grad():
        out = model(wav_tensor, None)

    framewise = torch.sigmoid(out["framewise_output"][0]).cpu().numpy()  # (T, 527)
    target_prob = framewise[:, class_id]  # (T,)

    # 5. Onset–offset detection
    frame_hop = 320 / 32000  # 10 ms = 0.01 sec
    segments = extract_onset_offset(target_prob, threshold, frame_hop=frame_hop)

    if debug:
        print("\n=== Detected Events ===")
        if len(segments) == 0:
            print("No segments detected.")
        for s, e in segments:
            print(f"  - {s:.2f}s  →  {e:.2f}s")

    # 6. Build temporal mask (binary)
    T = len(target_prob)
    mask = np.zeros(T)
    for s, e in segments:
        si = int(s / frame_hop)
        ei = int(e / frame_hop)
        mask[si:ei] = 1

    return segments, mask


# =====================================================
# ========== Interpolation to AudioSep =================
# =====================================================
def interp_mask(mask, target_len):
    """
    mask: (T_panns,) binary
    target_len: waveform length (samples)
    Returns: (target_len,) mask
    """
    T = len(mask)
    x1 = np.linspace(0, 1, T)
    x2 = np.linspace(0, 1, target_len)
    return np.interp(x2, x1, mask)


# =====================================================
# ========== AudioSep + mask ===========================
# =====================================================
def run_audiosep(audio_path, text_prompt, output_path, device, threshold, debug=False):

    # Load audio
    wav = load_audio(audio_path, sr=32000)
    if debug:
        print(f"[DEBUG] audio_len={len(wav)/32000:.2f}s")

    # Run PANNs SED
    segments, mask = run_panns_sed(wav, device, text_prompt, threshold, debug=debug)

    # Build AudioSep
    model = build_audiosep(
        config_yaml="./AudioSep/config/audiosep_base.yaml",
        checkpoint_path="./AudioSep/checkpoint/audiosep_base_4M_steps.ckpt",
        device=device
    )

    mixture = torch.tensor(wav).float()[None, None].to(device)

    conditions = model.query_encoder.get_query_embed(
        modality="text",
        text=[text_prompt],
        device=device
    )

    # Convert mask to waveform-length mask
    interp = interp_mask(mask, len(wav))
    # Wav mask must be numeric tensor
    mask_tensor = torch.tensor(interp).float()[None, None].to(device)

    # Apply mask to mixture
    masked_mixture = mixture * mask_tensor

    # Run AudioSep
    input_dict = {
        "mixture": masked_mixture,
        "condition": conditions
    }

    with torch.no_grad():
        out = model.ss_model(input_dict)["waveform"]
        out = out.squeeze().cpu().numpy()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    import soundfile as sf
    sf.write(output_path, out, 32000)

    print(f"[INFO] Saved to {output_path}")


# =====================================================
# ========== Main =====================================
# =====================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="PANNs onset–offset threshold")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_audiosep(
        args.audio,
        args.text,
        args.output,
        device,
        args.threshold,
        debug=args.debug
    )
