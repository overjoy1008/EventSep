import os
import json

import torch
import numpy as np

import bigvgan


def get_available_checkpoint_keys(model, ckpt):
    print("==> Attemp to reload from %s" % ckpt)
    state_dict = torch.load(ckpt)["state_dict"]
    current_state_dict = model.state_dict()
    new_state_dict = {}
    for k in state_dict.keys():
        if (
            k in current_state_dict.keys()
            and current_state_dict[k].size() == state_dict[k].size()
        ):
            new_state_dict[k] = state_dict[k]
        else:
            print("==> WARNING: Skipping %s" % k)
    print(
        "%s out of %s keys are matched"
        % (len(new_state_dict.keys()), len(state_dict.keys()))
    )
    return new_state_dict

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def torch_version_orig_mod_remove(state_dict):
    new_state_dict = {}
    new_state_dict["generator"] = {}
    for key in state_dict["generator"].keys():
        if("_orig_mod." in key):
            new_state_dict["generator"][key.replace("_orig_mod.","")] = state_dict["generator"][key]
        else:
            new_state_dict["generator"][key] = state_dict["generator"][key]
    return new_state_dict

# def get_vocoder(config, device, mel_bins):

#     with open("src/bigvgan/config.json", "r") as f:
#         config = json.load(f)
#     config = bigvgan.AttrDict(config)
#     vocoder = bigvgan.BigVGAN(config)
#     print("Load bigvgan_generator_16k")
#     ckpt = torch.load("src/bigvgan/g_01000000")
#     vocoder.load_state_dict(ckpt["generator"])
#     vocoder.eval()
#     vocoder.remove_weight_norm()
#     vocoder.to(device)

#     return vocoder

def get_vocoder(config, device, mel_bins):

    # ====== Patch: Robust absolute path resolution ======
    ROOT = os.path.dirname(os.path.abspath(__file__))              # FlowSep/src/utilities
    FLOWSEP_ROOT = os.path.abspath(os.path.join(ROOT, "..", "..")) # FlowSep/
    BIGVGAN_DIR = os.path.join(FLOWSEP_ROOT, "src", "bigvgan")

    CONFIG_PATH = os.path.join(BIGVGAN_DIR, "config.json")
    CKPT_PATH   = os.path.join(BIGVGAN_DIR, "g_01000000")

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"[FlowSep ERROR] BigVGAN config not found: {CONFIG_PATH}")

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"[FlowSep ERROR] BigVGAN checkpoint not found: {CKPT_PATH}")

    # ======================================================

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    config = bigvgan.AttrDict(config)
    vocoder = bigvgan.BigVGAN(config)

    print("Load bigvgan_generator_16k")
    ckpt = torch.load(CKPT_PATH, map_location=device)
    vocoder.load_state_dict(ckpt["generator"])

    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)

    return vocoder



def vocoder_infer(mels, vocoder, lengths=None):
    with torch.no_grad():
        wavs = vocoder(mels).squeeze(1)

    wavs = (wavs.cpu().numpy() * 32768).astype("int16")

    if lengths is not None:
        wavs = wavs[:, :lengths]

    # wavs = [wav for wav in wavs]

    # for i in range(len(mels)):
    #     if lengths is not None:
    #         wavs[i] = wavs[i][: lengths[i]]

    return wavs
