import numpy as np
import scipy.signal as sps

def blend_ensemble(a, b, mode, rate=0.3, cutoff=4000, sr=32000):
    if mode == "audiosep":
        return a
    if mode == "flowsep":
        return b
    if mode == "ensemble_a":
        return rate * b + (1.0 - rate) * a

    nyq = sr / 2.0
    norm_cut = cutoff / nyq
    b_lp, a_lp = sps.butter(6, norm_cut, btype="low")
    b_hp, a_hp = sps.butter(6, norm_cut, btype="high")

    a_low = sps.filtfilt(b_lp, a_lp, a)
    b_high = sps.filtfilt(b_hp, a_hp, b)

    return a_low + b_high
