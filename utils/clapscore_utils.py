# utils/clapscore_utils.py

import os
import laion_clap


def load_clapscore_model(model_type: str, device: str = "cuda"):
    """
    EventSep용 CLAPScore backbone 선택 함수.
    AudioSep CLAP_Encoder는 더 이상 사용하지 않음.
    모든 모델은 laion_clap.CLAP_Module(enable_fusion=...)로 통일.
    """
    base = "clapscore_models"

    # ========== enable_fusion 매핑 ==========
    fusion_map = {
        "audiosep": False,
        "630k_best": False,
        "630k_audioset_best": False,
        "music_speech_audioset": False,
        "music_speech": False,

        "630k_fusion_best": True,
        "630k_audioset_fusion_best": True,
    }

    if model_type not in fusion_map:
        raise ValueError(f"Unknown clapscore_type: {model_type}")

    enable_fusion = fusion_map[model_type]

    # ========== 로드할 checkpoint 결정 ==========
    ckpt_name_map = {
        "audiosep": "630k-best.pt",   # audiosep → 가장 근접한 630k-best 사용
        "630k_best": "630k-best.pt",
        "630k_audioset_best": "630k-audioset-best.pt",
        "630k_fusion_best": "630k-fusion-best.pt",
        "630k_audioset_fusion_best": "630k-audioset-fusion-best.pt",

        # AudioSep 제공 ckpt들도 laion_clap로 강제 통일
        "music_speech_audioset": "music_speech_audioset_epoch_15_esc_89.98.pt",
        "music_speech": "music_speech_epoch_15_esc_89.25.pt",
    }

    ckpt_file = os.path.join(base, ckpt_name_map[model_type])

    # ========== laion_clap Module 로드 ==========
    model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
    model.load_ckpt(ckpt_file)   # ★ 모든 모델 로딩 통일
    return model.to(device)

def load_auto_clap_model(device="cuda"):
    """
    CLAPScore(text-audio) = fusion_best
    CLAPScoreA(audio-audio) = audio_best
    """
    base = "clapscore_models"

    # text-audio model
    m_text = laion_clap.CLAP_Module(enable_fusion=True)
    m_text.load_ckpt(os.path.join(base, "630k-fusion-best.pt"))
    m_text = m_text.to(device).eval()

    # audio-audio model
    m_audio = laion_clap.CLAP_Module(enable_fusion=False)
    m_audio.load_ckpt(os.path.join(base, "630k-audioset-best.pt"))
    m_audio = m_audio.to(device).eval()

    return m_text, m_audio
