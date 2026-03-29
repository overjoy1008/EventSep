# AudioSep/benchmark_clapscore.py

import os
import numpy as np

from evaluation.evaluate_audiocaps import AudioCapsEvaluator
from evaluation.evaluate_audioset import AudioSetEvaluator
from evaluation.evaluate_vggsound import VGGSoundEvaluator
from evaluation.evaluate_music import MUSICEvaluator
from evaluation.evaluate_esc50 import ESC50Evaluator
# from evaluation.evaluate_clotho import ClothoEvaluator

from models.clap_encoder import CLAP_Encoder
from utils import (
    load_ss_model,
    parse_yaml,
    get_mean_sdr_from_dict,
)

def eval_all(checkpoint_path, config_yaml="config/audiosep_base.yaml"):

    print("========== AudioSep Evaluation (with CLAPScore) ==========")

    # ---------------------------------------------------------
    # Load AudioSep model + CLAP encoder
    # ---------------------------------------------------------
    configs = parse_yaml(config_yaml)
    device = "cuda"

    print("Loading CLAP_Encoder ...")
    clap_encoder = CLAP_Encoder().eval()

    print("Loading AudioSep model ...")
    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=clap_encoder
    ).to(device)
    pl_model.eval()

    # ---------------------------------------------------------
    # Evaluators (모두 CLAPScore 계산 버전으로 수정된 상태여야 함)
    # ---------------------------------------------------------
    evaluators = {
        # "VGGSound":    VGGSoundEvaluator(),
        # "AudioCaps":   AudioCapsEvaluator(),
        "AudioSet":    AudioSetEvaluator(),
        # "ESC50":       ESC50Evaluator(),
        # "MUSIC":       MUSICEvaluator(),
        # "Clotho":      ClothoEvaluator(),
    }

    all_results = {}

    for name, evaluator in evaluators.items():
        print(f"\n---------- Evaluating {name} ----------")

        results = evaluator(pl_model)

        # AudioSet만 형식이 다름 → 결과 변환
        if name == "AudioSet":
            # 기대형식: {"sisdrs_dict":..., "sdris_dict":...}
            sdris = evaluator.get_median_metrics(results, "sdris_dict")
            sisdrs = evaluator.get_median_metrics(results, "sisdrs_dict")

            SDRi = get_mean_sdr_from_dict(sdris)
            SISDR = get_mean_sdr_from_dict(sisdrs)

            # CLAPScore는 AudioSet에서 per-class로 하기는 어려움 → skip or set None
            final = {
                "SDR": None,
                "SDRi": float(SDRi),
                "SISDR": float(SISDR),
                "CLAPScore": None,
                "CLAPScoreA": None,
            }
        else:
            final = {
                "SDR": float(results["SDR"]),
                "SDRi": float(results["SDRi"]),
                "SISDR": float(results["SISDR"]),
                "CLAPScore": float(results["CLAPScore"]),
                "CLAPScoreA": float(results["CLAPScoreA"]),
            }

        all_results[name] = final

        # Print results
        print("\n===== RESULTS: {} =====".format(name))
        for k, v in final.items():
            if v is None:
                print(f"{k}: N/A")
            else:
                print(f"{k}: {v:.4f}")

    # ---------------------------------------------------------
    # Save log
    # ---------------------------------------------------------
    os.makedirs("eval_logs", exist_ok=True)
    log_path = "eval_logs/eval_clapscore_results.txt"

    with open(log_path, "w") as f:
        for dataset, metrics in all_results.items():
            f.write(f"[{dataset}]\n")
            for k, v in metrics.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")

    print(f"\nAll results saved to {log_path}")
    print("=============== DONE ===============")

    return all_results


if __name__ == "__main__":
    eval_all(
        checkpoint_path="checkpoint/audiosep_base_4M_steps.ckpt",
        config_yaml="config/audiosep_base.yaml"
    )
