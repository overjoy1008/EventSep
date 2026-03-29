import os
import datetime
import json
import logging
import librosa
import pickle
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import yaml
# from models.audiosep import AudioSep, get_model_class # 👈 제거: AudioSep 모델 종속성

# 참고: load_pretrained_panns에서 사용되는 Cnn14 등은 FlowSep에서 필요 없으므로,
# 해당 함수와 함께 관련된 외부 모델 임포트는 제거합니다.


def ignore_warnings():
    import warnings
    # Ignore UserWarning from torch.meshgrid
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.functional')

    # Refined regex pattern to capture variations in the warning message
    pattern = r"Some weights of the model checkpoint at roberta-base were not used when initializing RobertaModel: \['lm_head\..*'\].*"
    warnings.filterwarnings('ignore', message=pattern)


def create_logging(log_dir, filemode):
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


def float32_to_int16(x: float) -> int:
    x = np.clip(x, a_min=-1, a_max=1)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: int) -> float:
    return (x / 32767.0).astype(np.float32)


def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


# def get_audioset632_id_to_lb(ontology_path: str) -> Dict: # 👈 제거: AudioSet 632 클래스 특정 함수
#     r"""Get AudioSet 632 classes ID to label mapping."""
#     
#     audioset632_id_to_lb = {}
# 
#     with open(ontology_path) as f:
#         data_list = json.load(f)
# 
#     for e in data_list:
#         audioset632_id_to_lb[e["id"]] = e["name"]
# 
#     return audioset632_id_to_lb


# def load_pretrained_panns(...): # 👈 제거: AudioSep/PANNs 모델 종속성
# ...
#     return model


def energy(x):
    return torch.mean(x ** 2)


def magnitude_to_db(x):
    eps = 1e-10
    return 20. * np.log10(max(x, eps))


def db_to_magnitude(x):
    return 10. ** (x / 20)


def ids_to_hots(ids, classes_num, device):
    hots = torch.zeros(classes_num).to(device)
    for id in ids:
        hots[id] = 1
    return hots


def calculate_sdr(
    ref: np.ndarray,
    est: np.ndarray,
    eps=1e-10
) -> float:
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    reference = ref
    noise = est - reference


    numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)

    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)

    sdr = 10. * np.log10(numerator / denominator)

    return sdr


def calculate_sisdr(ref, est):
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """

    eps = np.finfo(ref.dtype).eps

    reference = ref.copy()
    estimate = est.copy()
    
    reference = reference.reshape(reference.size, 1)
    estimate = estimate.reshape(estimate.size, 1)

    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

    e_true = a * reference
    e_res = estimate - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    sisdr = 10 * np.log10((eps+ Sss)/(eps + Snn))

    return sisdr 

