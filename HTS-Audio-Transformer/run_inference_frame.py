# run_inference.py
from model.htsat import HTSAT_Swin_Transformer
import torch
import librosa
import numpy as np
import pandas as pd
import config

SAMPLE_RATE = 16000


def load_model():
    """Load HTSAT model + class labels (same logic as Cog setup())"""

    # Load class index mapping
    df = pd.read_csv("class_label_indice.csv", sep=",")
    idx_2_label = {}
    for row in df.iterrows():
        idx, _, label = row[1]
        idx_2_label[idx] = label

    # Load checkpoint
    checkpoints = torch.load(
        "./checkpoints/HTSAT_AudioSet_Saved_1.ckpt", map_location=torch.device("cpu")
    )
    new_checkpoints = {"state_dict": {}}
    for old_key in checkpoints["state_dict"].keys():
        new_key = old_key.replace("sed_model.", "")
        new_checkpoints["state_dict"][new_key] = checkpoints["state_dict"][old_key]

    # Init model
    sed_model = HTSAT_Swin_Transformer(
        spec_size=config.htsat_spec_size,
        patch_size=config.htsat_patch_size,
        in_chans=1,
        num_classes=config.classes_num,
        window_size=config.htsat_window_size,
        config=config,
        depths=config.htsat_depth,
        embed_dim=config.htsat_dim,
        patch_stride=config.htsat_stride,
        num_heads=config.htsat_num_head,
    )
    sed_model.load_state_dict(new_checkpoints["state_dict"])
    sed_model.eval()

    return sed_model, idx_2_label


def predict(audio_path, sed_model, idx_2_label):
    """Run inference with both clipwise & framewise SED"""

    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # HTSAT은 고정 입력 크기(약 10초) 필요 → 길이 제한
    max_len = SAMPLE_RATE * 10  # 160000 samples
    if len(waveform) > max_len:
        waveform = waveform[:max_len]
    elif len(waveform) < max_len:
        pad_len = max_len - len(waveform)
        waveform = np.pad(waveform, (0, pad_len), mode="constant")

    x = torch.from_numpy(waveform).float()

    with torch.no_grad():
        output_dict = sed_model(x[None, :], None, True)

        # ---------------- Clipwise (기존 Top-3) ----------------
        clip = output_dict["clipwise_output"]
        clip_post = torch.sigmoid(clip)[0].cpu().numpy()
        top3_idx = np.argsort(clip_post)[-3:][::-1]
        top3 = [
            [lbl, idx_2_label[lbl], float(clip_post[lbl])]
            for lbl in top3_idx
        ]

        # ---------------- Framewise (시간 기반 SED 추가) ----------------
        frame = output_dict["framewise_output"]  # (1, T, C)
        frame = torch.sigmoid(frame)[0].cpu().numpy()  # (T, C)
        T, C = frame.shape

        frame_hop = 10.0 / T  # seconds per frame
        times = np.arange(T) * frame_hop

        # Thresholding
        threshold = 0.5
        events = []
        for c in range(C):
            active = np.where(frame[:, c] > threshold)[0]
            if len(active) > 0:
                events.append((
                    idx_2_label[c],
                    times[active].tolist()
                ))

    return top3, events


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()

    print("Loading model...")
    sed_model, idx_2_label = load_model()

    print(f"Running inference on: {args.audio}")
    top3, events = predict(args.audio, sed_model, idx_2_label)

    # ---------------- Clipwise 출력 ----------------
    print("\nTop-3 Predictions:")
    for r in top3:
        print(f"{r[0]:4d} | {r[1]:20s} | {r[2]:.4f}")

    # ---------------- Framewise 출력 ----------------
    print("\nFramewise Event Timeline (threshold=0.5):")
    if len(events) == 0:
        print("No frame-level events detected.")
    else:
        for label, tlist in events:
            # 초 단위 시간 리스트 앞 10개만 보여주기
            print(f"{label:20s}: {tlist[:10]} ...")
