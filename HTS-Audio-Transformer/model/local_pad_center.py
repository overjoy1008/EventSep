# librosa==0.8.1

import numpy as np

def local_pad_center(data, size, axis=-1, **kwargs):
    """Pad an array to a target length along a target axis.

    librosa==0.8.1 version of pad_center, required by torchlibrosa==0.0.9.
    """

    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise ValueError(
            ("Target size ({:d}) must be "
             "at least input size ({:d})").format(size, n)
        )

    return np.pad(data, lengths, **kwargs)
