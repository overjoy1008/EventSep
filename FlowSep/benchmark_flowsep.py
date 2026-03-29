# # FlowSep/benchmark.py

# from evaluation.evaluate_vggsound import VGGSoundEvaluator
# from utils_flowsep import load_flowsep_model


# def eval(checkpoint_path, config_yaml):

#     print("------- FlowSep Evaluation (CLAPScore, CLAPScoreA, FAD) -------")

#     evaluator = VGGSoundEvaluator()

#     pl_model = load_flowsep_model(
#         config_yaml=config_yaml,
#         checkpoint_path=checkpoint_path,
#         device="cuda"
#     )

#     results = evaluator(pl_model)

#     print("\n===== RESULTS =====")
#     for k, v in results.items():
#         print(f"{k}: {v:.4f}")


# if __name__ == "__main__":
#     eval(
#         checkpoint_path="model_logs/pretrained/v2_100k.ckpt",
#         config_yaml="lass_config/2channel_flow.yaml"
#     )



# FlowSep/benchmark.py

import torch

# from evaluation.evaluate_vggsound import VGGSoundEvaluator
from evaluation.evaluate_audiocaps import AudioCapsEvaluator
# from evaluation.evaluate_audioset import AudioSetEvaluator
from evaluation.evaluate_esc50 import ESC50Evaluator
from evaluation.evaluate_music import MUSICEvaluator

from utils_flowsep import load_flowsep_model


def eval_all(checkpoint_path, config_yaml):

    print("============== FlowSep Evaluation ==============")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config     : {config_yaml}")
    print("==============================================\n")

    # ---------------------------------------------
    # Load FlowSep model (single model for all datasets)
    # ---------------------------------------------
    pl_model = load_flowsep_model(
        config_yaml=config_yaml,
        checkpoint_path=checkpoint_path,
        device="cuda"
    )
    pl_model.eval()

    # ---------------------------------------------
    # Evaluators to run (Clotho excluded)
    # ---------------------------------------------
    evaluators = {
        # "VGGSound":    VGGSoundEvaluator(),
        "AudioCaps":   AudioCapsEvaluator(),
        # "AudioSet":    AudioSetEvaluator(),
        # "ESC50":       ESC50Evaluator(),
        # "MUSIC":       MUSICEvaluator(),
    }

    all_results = {}

    # ---------------------------------------------
    # Run each evaluator
    # ---------------------------------------------
    for name, evaluator in evaluators.items():
        print(f"\n---------- Evaluating {name} ----------")
        results = evaluator(pl_model)
        all_results[name] = results

        print("\n===== RESULTS: {} =====".format(name))
        for k, v in results.items():
            print(f"{k}: {float(v):.4f}")

    print("\n============== ALL EVALUATIONS DONE ==============")

    return all_results


# ---------------------------------------------
# Main
# ---------------------------------------------
if __name__ == "__main__":
    eval_all(
        checkpoint_path="model_logs/pretrained/v2_100k.ckpt",
        config_yaml="lass_config/2channel_flow.yaml"
    )
