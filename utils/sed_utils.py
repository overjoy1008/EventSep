import numpy as np
import torch

def extract_mask_from_panns(panns, mixture, cid, threshold=0.5, device="cuda"):
    wav_tensor = torch.tensor(mixture).float()[None].to(device)

    with torch.no_grad():
        out = panns(wav_tensor, None)

    framewise = torch.sigmoid(out["framewise_output"][0]).cpu().numpy()
    p = framewise[:, cid]

    mask = np.zeros_like(p)
    mask[p >= threshold] = 1.0
    mask[p < threshold] = p[p < threshold] / threshold

    x1 = np.linspace(0, 1, len(mask))
    x2 = np.linspace(0, 1, len(mixture))
    return np.interp(x2, x1, mask)
