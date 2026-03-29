# FlowSep/utils_flowsep.py

import sys
import os
import yaml
import torch

# FlowSep/src 를 PYTHONPATH에 추가
SRC = os.path.join(os.path.dirname(__file__), "src")
if SRC not in sys.path:
    sys.path.append(SRC)

from latent_diffusion.util import instantiate_from_config

from query_encoder_wrapper import FlowSepQueryEncoder
from ss_model_wrapper import FlowSepSeparationWrapper
from pl_model_wrapper import FlowSepLightningLike


def load_flowsep_model(config_yaml, checkpoint_path, device="cuda"):
    """
    AudioSep의 load_ss_model()과 동일한 역할을 하는 FlowSep용 로더.
    - latent diffusion 모델 로드
    - checkpoint 로딩
    - FlowSepSeparationWrapper + Dummy QueryEncoder 로 감싸서
      AudioSep evaluator가 기대하는 pl_model 인터페이스로 변환
    """

    # 전체 config 로드 (AudioDataset에도 그대로 넘김)
    configs = yaml.load(open(config_yaml, "r"), Loader=yaml.FullLoader)

    # latent diffusion 모델 인스턴스
    model = instantiate_from_config(configs["model"])
    ckpt = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.to(device).eval()

    # QueryEncoder는 FlowSep에서는 실제로 안 쓰이므로 dummy
    query_encoder = FlowSepQueryEncoder()

    # ss_model wrapper는 configs를 받아 AudioDataset을 내부에서 생성
    ss_model = FlowSepSeparationWrapper(model, configs, device)

    # AudioSep LightningModule 비슷하게 감싸기
    pl_like = FlowSepLightningLike(
        ss_model=ss_model,
        query_encoder=query_encoder,
        device=device
    )

    return pl_like
