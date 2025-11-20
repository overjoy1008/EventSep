import os
import csv
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm
import librosa
import lightning.pytorch as pl

from models.clap_encoder import CLAP_Encoder
from utils import calculate_sdr, calculate_sisdr


class ESC50Evaluator:
    def __init__(self, sampling_rate: int = 32000) -> None:
        r"""ESC-50 evaluator."""

        self.sampling_rate = sampling_rate

        with open("evaluation/metadata/esc50_eval.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            eval_list = [row for row in csv_reader][1:]

        self.eval_list = eval_list
        self.audio_dir = "evaluation/data/esc50"

        print("Loading CLAP model for ESC-50 evaluation ...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clap = CLAP_Encoder().eval().to(self.device)

    def __call__(self, pl_model: pl.LightningModule) -> Dict:
        r"""Evaluate."""

        print("Evaluation on ESC-50 with [text label] queries.")

        pl_model.eval()
        device = pl_model.device

        sdr_list, sdri_list, sisdr_list = [], [], []
        clap_score_list, clapA_score_list = [], []

        with torch.no_grad():
            for idx, caption, _, _ in tqdm(self.eval_list):
                source_path = os.path.join(self.audio_dir, f"segment-{idx}.wav")
                mixture_path = os.path.join(self.audio_dir, f"mixture-{idx}.wav")

                if not (os.path.exists(source_path) and os.path.exists(mixture_path)):
                    continue

                source, _ = librosa.load(
                    source_path, sr=self.sampling_rate, mono=True
                )
                mixture, _ = librosa.load(
                    mixture_path, sr=self.sampling_rate, mono=True
                )

                sdr_no_sep = calculate_sdr(ref=source, est=mixture)
                sisdr_no_sep = calculate_sisdr(ref=source, est=mixture)

                text_query = caption

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
