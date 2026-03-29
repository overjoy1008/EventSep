# FlowSep/pl_model_wrapper.py

import torch.nn as nn


class FlowSepLightningLike(nn.Module):
    """
    AudioSep evaluator가 기대하는 최소 인터페이스:
        - pl_model.ss_model(input_dict)
        - pl_model.query_encoder
        - pl_model.device
    """

    def __init__(self, ss_model, query_encoder, device):
        super().__init__()
        self.ss_model = ss_model
        self.query_encoder = query_encoder
        self._device = device

    @property
    def device(self):
        return self._device

    def forward(self, *args, **kwargs):
        raise NotImplementedError
