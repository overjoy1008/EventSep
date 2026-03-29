# htsat_infer.py
import torch
import librosa
import numpy as np
import csv

import config as cfg               # HTS 레포의 config.py
from model.htsat import HTSAT_Swin_Transformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 모델 로드
def load_htsat(checkpoint_path: str):
    # config.py 안에서 classes_num, sample_rate, window_size, hop_size, mel_bins 등을 읽어 씀
    model = HTSAT_Swin_Transformer(
        config=cfg,
        num_classes=cfg.classes_num,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    # Lightning ckpt라면 보통 "state_dict" 키에 가중치가 들어 있음
    state_dict = ckpt.get("state_dict", ckpt)
    # lightning prefix가 붙어 있다면 여기서 정리해줘야 할 수도 있음
    new_state_dict = {}
    for k, v in state_dict.items():
        # 예: "model." 같은 prefix 제거가 필요하면 여기서 처리
        if k.startswith("model."):
            new_state_dict[k[len("model."):]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model

# 2. 오디오 로드 + HTS 입력 형식으로 변환
def load_audio(path: str):
    wav, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (B=1, C=1, T)
    return wav.to(DEVICE)

# 3. 클래스 인덱스 → 라벨 이름 매핑 (class_label_indice.csv 사용)
def load_label_map(csv_path="class_label_indice.csv"):
    # CSV 형식은 레포 안 파일 기준 (index, mid, display_name) 형태라고 가정
    labels = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["display_name"])
    return labels

def infer(model, audio_tensor, top_k=5):
    with torch.no_grad():
        # infer_mode=True로 길이 유연하게 처리
        out = model(audio_tensor, infer_mode=True)

    clipwise = out["clipwise_output"][0].cpu().numpy()    # (527,)
    framewise = out["framewise_output"][0].cpu().numpy()  # (T, 527) 형태

    # 상위 top_k 클래스 뽑기
    top_idx = np.argsort(-clipwise)[:top_k]
    return clipwise, framewise, top_idx

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/HTSAT_AudioSet_Saved_1.ckpt")
    parser.add_argument("--audio", type=str,
                        default="input/lullaby_short.mp3")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    labels = load_label_map("class_label_indice.csv")
    model = load_htsat(args.checkpoint)
    audio = load_audio(args.audio)

    clipwise, framewise, top_idx = infer(model, audio, top_k=args.topk)

    print("Top-{} predictions:".format(args.topk))
    for i in top_idx:
        print(f"{i:3d} | {labels[i] if i < len(labels) else 'UNK'} | {clipwise[i]:.4f}")
