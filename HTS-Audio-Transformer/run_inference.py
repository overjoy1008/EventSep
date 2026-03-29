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
        # Remove "sed_model." prefix to match model state_dict
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
    """Run inference (same logic as Cog predict())"""

    waveform, sr = librosa.load(audio_path, sr=SAMPLE_RATE)

    # HTSAT은 고정 입력 크기(약 10초) 필요 → 길이 제한
    max_len = SAMPLE_RATE * 10  # 160000 samples
    if len(waveform) > max_len:
        waveform = waveform[:max_len]
    elif len(waveform) < max_len:
        # pad to exactly 10 sec
        pad_len = max_len - len(waveform)
        waveform = np.pad(waveform, (0, pad_len), mode="constant")

    x = torch.from_numpy(waveform).float()


    with torch.no_grad():
        output_dict = sed_model(x[None, :], None, True)
        pred = output_dict["clipwise_output"]
        pred_post = torch.sigmoid(pred)[0].cpu().numpy()

    pred_labels = np.argsort(pred_post)[-3:][::-1]

    return [
        [lbl, idx_2_label[lbl], float(pred_post[lbl])]
        for lbl in pred_labels
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    args = parser.parse_args()

    print("Loading model...")
    sed_model, idx_2_label = load_model()

    print(f"Running inference on: {args.audio}")
    results = predict(args.audio, sed_model, idx_2_label)

    print("\nTop-3 Predictions:")
    for r in results:
        print(f"{r[0]:4d} | {r[1]:20s} | {r[2]:.4f}")
