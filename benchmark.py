# # ========================== benchmark.py ==========================

# import os
# import sys
# import datetime

# sys.path.append(os.path.join(os.getcwd(), "AudioSep"))
# from AudioSep.utils import load_ss_model, parse_yaml

# sys.path.append(os.path.join(os.getcwd(), "FlowSep/src"))
# sys.path.append(os.path.join(os.getcwd(), "FlowSep"))
# from utils_flowsep import load_flowsep_model

# from evaluation.evaluate_vggsound_eventsep import VGGSoundEventSepEvaluator


# def benchmark(
#     audiosep_ckpt="AudioSep/checkpoint/audiosep_base_4M_steps.ckpt",
#     audiosep_config="AudioSep/config/audiosep_base.yaml",
#     flowsep_ckpt="FlowSep/model_logs/pretrained/v2_100k.ckpt",
#     flowsep_config="FlowSep/lass_config/2channel_flow.yaml",
#     ensemble="audiosep",
#     sed_mask="soft",
#     sed_threshold=0.5,
#     ensemble_rate=0.3,
#     ensemble_freq=4000,
#     clapscore_type="audiosep",
#     embedding_model_type="minilm",
#     demo_mode=False,
#     use_demucs=False,
# ):

#     device = "cuda"

#     configs = parse_yaml(audiosep_config)
#     from models.clap_encoder import CLAP_Encoder
#     query_encoder = CLAP_Encoder().eval()

#     pl_audiosep = load_ss_model(
#         configs=configs,
#         checkpoint_path=audiosep_ckpt,
#         query_encoder=query_encoder
#     ).to(device)

#     pl_flowsep = load_flowsep_model(
#         config_yaml=flowsep_config,
#         checkpoint_path=flowsep_ckpt,
#         device=device,
#     )

#     evaluator = VGGSoundEventSepEvaluator(
#         sampling_rate=32000,
#         sed_threshold=sed_threshold,
#         mask_mode=sed_mask,
#         ensemble=ensemble,
#         ensemble_rate=ensemble_rate,
#         ensemble_freq=ensemble_freq,
#         clapscore_type=clapscore_type,
#         embedding_model_type=embedding_model_type,
#         demo_mode=demo_mode,
#         use_demucs=use_demucs,
#     )

#     results = evaluator(pl_audiosep, pl_flowsep)

#     print("\n===== FINAL RESULT =====")
#     for k, v in results.items():
#         print(f"{k}: {v:.4f}")

#     return results

# def save_results_to_txt(results, save_dir, exp_name, params):
#     os.makedirs(save_dir, exist_ok=True)

#     # 현재 시각 기반 파일명
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"{exp_name}_{timestamp}.txt"
#     filepath = os.path.join(save_dir, filename)

#     with open(filepath, "w") as f:
#         f.write("===== Experiment Parameters =====\n")
#         for k, v in params.items():
#             f.write(f"{k}: {v}\n")

#         f.write("\n===== Benchmark Results =====\n")
#         for k, v in results.items():
#             f.write(f"{k}: {v:.4f}\n")

#     print(f"[Saved] → {filepath}")
#     return filepath


# if __name__ == '__main__':
#     EXPERIMENTS = [
#         dict(
#             ensemble="audiosep",
#             sed_mask="none",
#             sed_threshold=0.55,
#             ensemble_rate=0.1,
#             ensemble_freq=4000,
#             clapscore_type="auto",
#             embedding_model_type="minilm",
#             demo_mode=True,
#             use_demucs=True,
#         ),
#         dict(
#             ensemble="audiosep",
#             sed_mask="none",
#             sed_threshold=0.55,
#             ensemble_rate=0.1,
#             ensemble_freq=4000,
#             clapscore_type="auto",
#             embedding_model_type="minilm",
#             demo_mode=True,
#             use_demucs=False,
#         ),
#     ]

#     easy_code = "DMND"
#     repository = ""
#     for idx, params in enumerate(EXPERIMENTS, start=1):
#         print(f"\n======= Running experiment #{idx} =======")
#         print(params)
#         results = benchmark(**params)
#         save_results_to_txt(
#             results=results,
#             save_dir=f"./results/{repository}",
#             exp_name=f"{easy_code}_{idx}",
#             params=params
#         )


# ========================== benchmark.py ==========================

import os
import sys
import datetime

# AudioSep
sys.path.append(os.path.join(os.getcwd(), "AudioSep"))
from AudioSep.utils import load_ss_model, parse_yaml

# FlowSep
sys.path.append(os.path.join(os.getcwd(), "FlowSep/src"))
sys.path.append(os.path.join(os.getcwd(), "FlowSep"))
from utils_flowsep import load_flowsep_model

# ===== Evaluators =====
from evaluation.evaluate_vggsound_eventsep import VGGSoundEventSepEvaluator
from evaluation.evaluate_audiocaps_eventsep import AudioCapsEvaluator
from evaluation.evaluate_audioset_eventsep import AudioSetEvaluator
from evaluation.evaluate_esc50_eventsep import ESC50Evaluator
from evaluation.evaluate_music_eventsep import MUSICEvaluator


# =====================================================================
#                          Unified Benchmark
# =====================================================================

DATASET_EVALUATORS = {
    "vggsound": VGGSoundEventSepEvaluator,
    "audiocaps": AudioCapsEvaluator,
    "audioset": AudioSetEvaluator,
    "esc50": ESC50Evaluator,
    "music": MUSICEvaluator,
}


def benchmark(
    dataset="vggsound",  # ★ 추가됨
    audiosep_ckpt="AudioSep/checkpoint/audiosep_base_4M_steps.ckpt",
    audiosep_config="AudioSep/config/audiosep_base.yaml",
    flowsep_ckpt="FlowSep/model_logs/pretrained/v2_100k.ckpt",
    flowsep_config="FlowSep/lass_config/2channel_flow.yaml",
    ensemble="audiosep",
    sed_mask="soft",
    sed_threshold=0.5,
    ensemble_rate=0.3,
    ensemble_freq=4000,
    clapscore_type="audiosep",
    embedding_model_type="minilm",
    demo_mode=False,
    use_demucs=False,
):

    assert (
        dataset in DATASET_EVALUATORS
    ), f"[ERROR] Unknown dataset '{dataset}'. Choose from: {list(DATASET_EVALUATORS.keys())}"

    device = "cuda"

    # ========== Load AudioSep ==========
    configs = parse_yaml(audiosep_config)
    from models.clap_encoder import CLAP_Encoder

    query_encoder = CLAP_Encoder().eval()

    pl_audiosep = load_ss_model(
        configs=configs, checkpoint_path=audiosep_ckpt, query_encoder=query_encoder
    ).to(device)

    # ========== Load FlowSep ==========
    pl_flowsep = load_flowsep_model(
        config_yaml=flowsep_config,
        checkpoint_path=flowsep_ckpt,
        device=device,
    )

    # ========== Create evaluator ==========
    EvaluatorClass = DATASET_EVALUATORS[dataset]

    evaluator = EvaluatorClass(
        sampling_rate=32000,
        sed_threshold=sed_threshold,
        mask_mode=sed_mask,
        ensemble=ensemble,
        ensemble_rate=ensemble_rate,
        ensemble_freq=ensemble_freq,
        clapscore_type=clapscore_type,
        embedding_model_type=embedding_model_type,
        demo_mode=demo_mode,
        use_demucs=use_demucs,
    )

    # ========== Run Evaluation ==========
    results = evaluator(pl_audiosep, pl_flowsep)

    print("\n===== FINAL RESULT =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results


# =====================================================================
#                         Save Result to TXT
# =====================================================================


def save_results_to_txt(results, save_dir, exp_name, params):
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{exp_name}_{timestamp}.txt"
    filepath = os.path.join(save_dir, filename)

    with open(filepath, "w") as f:
        f.write("===== Experiment Parameters =====\n")
        for k, v in params.items():
            f.write(f"{k}: {v}\n")

        f.write("\n===== Benchmark Results =====\n")
        for k, v in results.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"[Saved] → {filepath}")
    return filepath


# =====================================================================
#                                MAIN
# =====================================================================

if __name__ == "__main__":

    demo_mode = False

    EXPERIMENTS = [
        dict(
            dataset="vggsound",
            ensemble="audiosep",
            sed_mask="hard",
            sed_threshold=0.55,
            ensemble_rate=0.1,
            ensemble_freq=4000,
            clapscore_type="auto",
            embedding_model_type="minilm",
            demo_mode=demo_mode,
            use_demucs=False,
        ),
        dict(
            dataset="vggsound",
            ensemble="audiosep",
            sed_mask="hard",
            sed_threshold=0.6,
            ensemble_rate=0.1,
            ensemble_freq=4000,
            clapscore_type="auto",
            embedding_model_type="minilm",
            demo_mode=demo_mode,
            use_demucs=False,
        ),
        dict(
            dataset="vggsound",
            ensemble="audiosep",
            sed_mask="hard",
            sed_threshold=0.5,
            ensemble_rate=0.1,
            ensemble_freq=4000,
            clapscore_type="auto",
            embedding_model_type="minilm",
            demo_mode=demo_mode,
            use_demucs=False,
        ),
        dict(
            dataset="vggsound",
            ensemble="audiosep",
            sed_mask="hard",
            sed_threshold=0.65,
            ensemble_rate=0.1,
            ensemble_freq=4000,
            clapscore_type="auto",
            embedding_model_type="minilm",
            demo_mode=demo_mode,
            use_demucs=False,
        ),
        # dict(
        #     dataset="esc50",
        #     ensemble="flowsep",
        #     sed_mask="none",
        #     sed_threshold=0.55,
        #     ensemble_rate=0.1,
        #     ensemble_freq=4000,
        #     clapscore_type="auto",
        #     embedding_model_type="minilm",
        #     demo_mode=demo_mode,
        #     use_demucs=False,
        # ),
        # dict(
        #     dataset="esc50",
        #     ensemble="audiosep",
        #     sed_mask="soft",
        #     sed_threshold=0.55,
        #     ensemble_rate=0.1,
        #     ensemble_freq=4000,
        #     clapscore_type="auto",
        #     embedding_model_type="minilm",
        #     demo_mode=demo_mode,
        #     use_demucs=False,
        # ),
        # dict(
        #     dataset="esc50",
        #     ensemble="ensemble-a",
        #     sed_mask="soft",
        #     sed_threshold=0.55,
        #     ensemble_rate=0.1,
        #     ensemble_freq=4000,
        #     clapscore_type="auto",
        #     embedding_model_type="minilm",
        #     demo_mode=demo_mode,
        #     use_demucs=False,
        # ),
        # dict(
        #     dataset="esc50",
        #     ensemble="ensemble-b",
        #     sed_mask="soft",
        #     sed_threshold=0.55,
        #     ensemble_rate=0.1,
        #     ensemble_freq=4000,
        #     clapscore_type="auto",
        #     embedding_model_type="minilm",
        #     demo_mode=demo_mode,
        #     use_demucs=False,
        # ),
        # dict(
        #     dataset="esc50",
        #     ensemble="ensemble-c",
        #     sed_mask="soft",
        #     sed_threshold=0.55,
        #     ensemble_rate=0.1,
        #     ensemble_freq=4000,
        #     clapscore_type="auto",
        #     embedding_model_type="minilm",
        #     demo_mode=demo_mode,
        #     use_demucs=False,
        # ),
        # dict(
        #     dataset="esc50",
        #     ensemble="ensemble-d",
        #     sed_mask="soft",
        #     sed_threshold=0.55,
        #     ensemble_rate=0.1,
        #     ensemble_freq=4000,
        #     clapscore_type="auto",
        #     embedding_model_type="minilm",
        #     demo_mode=demo_mode,
        #     use_demucs=False,
        # ),
        # dict(
        #     dataset="esc50",
        #     ensemble="ensemble-e",
        #     sed_mask="soft",
        #     sed_threshold=0.55,
        #     ensemble_rate=0.1,
        #     ensemble_freq=4000,
        #     clapscore_type="auto",
        #     embedding_model_type="minilm",
        #     demo_mode=demo_mode,
        #     use_demucs=False,
        # ),
    ]

    easy_code = "DMND"
    repository = "hard_mask_experiments"

    for idx, params in enumerate(EXPERIMENTS, start=1):
        print(f"\n======= Running experiment #{idx} =======")
        print(params)
        results = benchmark(**params)
        save_results_to_txt(
            results=results,
            save_dir=f"./results/{repository}",
            exp_name=f"{easy_code}_{idx}",
            params=params,
        )
