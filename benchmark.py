import os
from tqdm import tqdm
import numpy as np
# from evaluation.evaluate_audioset import AudioSetEvaluator
# from evaluation.evaluate_audiocaps import AudioCapsEvaluator
from evaluation.evaluate_vggsound import VGGSoundEvaluator
# from evaluation.evaluate_music import MUSICEvaluator
# from evaluation.evaluate_esc50 import ESC50Evaluator
# from evaluation.evaluate_clotho import ClothoEvaluator
from models.clap_encoder import CLAP_Encoder

from utils import (
    load_ss_model,
    calculate_sdr,
    calculate_sisdr,
    parse_yaml,
    get_mean_sdr_from_dict,
)


def eval(checkpoint_path, config_yaml='AudioSep/config/audiosep_base.yaml'):

    log_dir = 'eval_logs'
    os.makedirs(log_dir, exist_ok=True)

    device = "cuda"
    
    configs = parse_yaml(config_yaml)

    # # AudioSet Evaluators
    # audioset_evaluator = AudioSetEvaluator()
    # # AudioCaps Evaluator
    # audiocaps_evaluator = AudioCapsEvaluator()
    # VGGSound+ Evaluator (SED-guided EventSep 버전)
    vggsound_evaluator = VGGSoundEvaluator(sampling_rate=32000, sed_threshold=0.3, use_sed_mask=True)
    # # Clotho Evaluator
    # clotho_evaluator = ClothoEvaluator()
    # # MUSIC Evaluator
    # music_evaluator = MUSICEvaluator()
    # # ESC-50 Evaluator
    # esc50_evaluator = ESC50Evaluator()
    
    # Load model
    query_encoder = CLAP_Encoder().eval()

    pl_model = load_ss_model(
        configs=configs,
        checkpoint_path=checkpoint_path,
        query_encoder=query_encoder
    ).to(device)

    print(f'-------  Start Evaluation  -------')

    # ===== VGGSound+ (SDRi / SISDR / CLAPScore) =====
    vgg_results = vggsound_evaluator(pl_model)
    msg_vgg = (
        "VGGSound Avg SDRi: {:.3f}, SISDR: {:.3f}, CLAPScore: {:.3f}, CLAPScoreA: {:.3f}"
        .format(
            vgg_results["SDRi"],
            vgg_results["SISDR"],
            vgg_results["CLAPScore"],
            vgg_results["CLAPScoreA"],
        )
    )
    print(msg_vgg)
    
    # # ===== MUSIC =====
    # SISDR, SDRi = music_evaluator(pl_model)
    # msg_music = "MUSIC Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    # print(msg_music)

    # # ===== ESC-50 =====
    # SISDR, SDRi = esc50_evaluator(pl_model)
    # msg_esc50 = "ESC-50 Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    # print(msg_esc50)

    # # ===== AudioSet =====
    # stats_dict = audioset_evaluator(pl_model=pl_model)
    # median_sdris = {}
    # median_sisdrs = {}

    # for class_id in range(527):
    #     median_sdris[class_id] = np.nanmedian(stats_dict["sdris_dict"][class_id])
    #     median_sisdrs[class_id] = np.nanmedian(stats_dict["sisdrs_dict"][class_id])

    # SDRi = get_mean_sdr_from_dict(median_sdris)
    # SISDR = get_mean_sdr_from_dict(median_sisdrs)
    # msg_audioset = "AudioSet Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    # print(msg_audioset)

    # # ===== AudioCaps =====
    # SISDR, SDRi = audiocaps_evaluator(pl_model)
    # msg_audiocaps = "AudioCaps Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    # print(msg_audiocaps)

    # # Clotho (필요하면 다시 활성화)
    # SISDR, SDRi = clotho_evaluator(pl_model)
    # msg_clotho = "Clotho Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    # print(msg_clotho)
    
    # msgs = [msg_audioset, msg_vgg, msg_audiocaps, msg_clotho, msg_music, msg_esc50]
    msgs = [msg_vgg]

    # open file in write mode
    log_path = os.path.join(log_dir, 'eval_results.txt')
    with open(log_path, 'w') as fp:
        for msg in msgs:
            fp.write(msg + '\n')
    print(f'Eval log is written to {log_path} ...')
    print('-------------------------  Done  ---------------------------')


if __name__ == '__main__':
    eval(checkpoint_path='AudioSep/checkpoint/audiosep_base_4M_steps.ckpt')
