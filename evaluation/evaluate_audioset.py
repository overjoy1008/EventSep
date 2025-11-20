import os
import re
from typing import Dict, List

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import librosa
import lightning.pytorch as pl

from models.clap_encoder import CLAP_Encoder
from utils import calculate_sdr, calculate_sisdr


meta_csv_file = "evaluation/metadata/class_labels_indices.csv"
df = pd.read_csv(meta_csv_file, sep=",")

IDS = df["mid"].tolist()
LABELS = df["display_name"].tolist()
CLASSES_NUM = len(LABELS)

IX_TO_LB = {i: label for i, label in enumerate(LABELS)}


class AudioSetEvaluator:
    def __init__(
        self,
        audios_dir: str = "evaluation/data/audioset",
        classes_num: int = 527,
        sampling_rate: int = 32000,
        number_per_class: int = 10,
    ) -> None:
        r"""AudioSet evaluator."""

        self.audios_dir = audios_dir
        self.classes_num = classes_num
        self.number_per_class = number_per_class
        self.sampling_rate = sampling_rate

        print("Loading CLAP model for AudioSet evaluation ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clap = CLAP_Encoder().eval().to(self.device)

    @torch.no_grad()
    def __call__(self, pl_model: pl.LightningModule) -> Dict:
        r"""Evaluate."""

        pl_model.eval()
        device = pl_model.device

        sdr_list, sdri_list, sisdr_list = [], [], []
        clap_score_list, clapA_score_list = [], []

        print("Evaluation on AudioSet with [text label] queries.")

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

                if not (
                    os.path.exists(source_path) and os.path.exists(mixture_path)
                ):
                    continue

                source, _ = librosa.load(
                    source_path, sr=self.sampling_rate, mono=True
                )
                mixture, _ = librosa.load(
                    mixture_path, sr=self.sampling_rate, mono=True
                )

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)
                sisdr_no_sep = calculate_sisdr(ref=source, est=mixture)

                text_query = IX_TO_LB[class_id]

                conditions = pl_model.query_encoder.get_query_embed(
                    modality="text",
                    text=[text_query],
                    device=device,
                )

                input_dict = {
                    "mixture": torch.tensor(mixture)[None, None, :].float().to(device),
                    "condition": conditions,
                }

                sep = pl_model.ss_model(input_dict)["waveform"]
                sep = sep.squeeze().cpu().numpy()

                L = min(len(source), len(sep), len(mixture))
                source = source[:L]
                sep = sep[:L]
                mixture_trim = mixture[:L]

                sdr_sep = calculate_sdr(ref=source, est=sep)
                sdri = sdr_sep - sdr_no_sep
                sisdr_sep = calculate_sisdr(ref=source, est=sep)
                _ = sisdr_no_sep

                sdr_list.append(sdr_sep)
                sdri_list.append(sdri)
                sisdr_list.append(sisdr_sep)

                # ---- CLAP ----
                sep_tensor = torch.tensor(sep, dtype=torch.float32).unsqueeze(0).to(
                    self.device
                )
                src_tensor = torch.tensor(source, dtype=torch.float32).unsqueeze(0).to(
                    self.device
                )

                sep_emb = self.clap._get_audio_embed(sep_tensor)
                text_emb = self.clap._get_text_embed([text_query])

                clap_score = torch.cosine_similarity(sep_emb, text_emb).item()
                clap_score_list.append(clap_score)

                src_emb = self.clap._get_audio_embed(src_tensor)
                clapA = torch.cosine_similarity(sep_emb, src_emb).item()
                clapA_score_list.append(clapA)

        return {
            "SDR": float(np.mean(sdr_list)),
            "SDRi": float(np.mean(sdri_list)),
            "SISDR": float(np.mean(sisdr_list)),
            "CLAPScore": float(np.mean(clap_score_list)),
            "CLAPScoreA": float(np.mean(clapA_score_list)),
        }

    def _get_audio_names(self, audios_dir: str) -> List[str]:
        audio_names = sorted(os.listdir(audios_dir))
        audio_names = [x for x in audio_names if ".wav" in x]
        audio_names = [
            re.search(r"(.*),(mixture|source)\.wav", x).group(1) for x in audio_names
        ]
        return sorted(list(set(audio_names)))
