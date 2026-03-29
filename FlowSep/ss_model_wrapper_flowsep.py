# FlowSep/ss_model_wrapper.py

import torch
import torch.nn as nn
import numpy as np

from utilities.data.dataset import AudioDataset


class FlowSepSeparationWrapper(nn.Module):
    """
    FlowSep latent diffusion 모델을
    AudioSep evaluator가 기대하는 ss_model 인터페이스로 감싼 wrapper.

    forward(input_dict) -> {"waveform": Tensor(1,1,T)}
    """

    def __init__(self, flowsep_model, configs, device):
        super().__init__()
        self.model = flowsep_model
        self.device = device

        # lass_inference.py에서 쓰던 AudioDataset 그대로 사용
        # feature 추출용으로만 쓴다.
        self.val_dataset = AudioDataset(configs, split="test", add_ons=[])

        # FlowSep 학습/인퍼런스가 16kHz 기준이라 고정
        self.sampling_rate = 16000

    def forward(self, input_dict):
        """
        input_dict:
            - "mixture": torch.Tensor (1, 1, T)
            - "text":    str (query text)
        """

        mixture = input_dict["mixture"]      # (1,1,T) on device
        text    = input_dict["text"]        # string

        # numpy (T,) 로 변환
        mixture_np = mixture.squeeze().detach().cpu().numpy().astype(np.float32)

        # lass_inference.py 에서처럼 feature 추출
        #   mixed_mel: (time, mel_bins)
        #   stft:      (time, freq)  (정확 shape는 dataset 구현에 따름)
        mixed_mel, stft = self.val_dataset.wav_feature_extraction(
            mixture_np.reshape(1, -1)
        )

        # numpy → tensor 변환 (혼합 타입 방지)
        if isinstance(mixed_mel, np.ndarray):
            mixed_mel_t = torch.from_numpy(mixed_mel).float()
        else:
            mixed_mel_t = mixed_mel.float()

        if isinstance(stft, np.ndarray):
            stft_t = torch.from_numpy(stft).float()
        else:
            stft_t = stft.float()


        # lass_inference.py에서 만든 batch 포맷 그대로 재현
        # ----------------------------------------------------
        # batch["fname"]         = [cur_wav]
        # batch["text"]          = [cur_text]
        # batch["caption"]       = [cur_text]
        # batch["waveform"]      = torch.rand(1,1,163840).cuda()
        # batch["log_mel_spec"]  = torch.rand(1,1024,64).cuda()
        # batch["sampling_rate"] = torch.tensor([16000]).cuda()
        # batch["label_vector"]  = torch.rand(1,527).cuda()
        # batch["stft"]          = torch.rand(1,1024,512).cuda()
        # noise_waveform, _      = val_dataset.read_wav_file(cur_wav)
        # noise_waveform         = noise_waveform[0][:163840]
        # mixed_mel, stft        = val_dataset.wav_feature_extraction(noise_waveform)
        # batch["mixed_waveform"]= torch.from_numpy(noise_waveform.reshape(1,1,163840))
        # batch["mixed_mel"]     = mixed_mel.reshape(1,T,F)
        # ----------------------------------------------------
        # 여기서는 source waveform 대신 mixture를 그대로 사용

        # 길이는 적당히 mixture 길이 그대로 사용
        T = mixture_np.shape[0]

        batch = {}
        batch["fname"]         = ["dummy.wav"]      # fname 필수
        batch["text"]          = [text]
        batch["caption"]       = [text]
        batch["waveform"]      = torch.from_numpy(mixture_np.reshape(1, 1, T)).float()
        batch["log_mel_spec"]  = mixed_mel_t.reshape(1, mixed_mel_t.shape[0], mixed_mel_t.shape[1])
        batch["sampling_rate"] = torch.tensor([self.sampling_rate])
        batch["label_vector"]  = torch.zeros(1, 527)   # shape만 맞으면 됨 (실제 학습에는 안씀)
        batch["stft"]          = stft_t.reshape(1, stft_t.shape[0], stft_t.shape[1])

        batch["mixed_waveform"] = torch.from_numpy(mixture_np.reshape(1, 1, T)).float()
        batch["mixed_mel"]      = mixed_mel_t.reshape(1, mixed_mel_t.shape[0], mixed_mel_t.shape[1])

        # FlowSep generate_sample() 호출
        # generate_sample 는 waveform (np.ndarray) 를 반환
        waveform = self.model.generate_sample(
            [batch],
            name="eval",
            unconditional_guidance_scale=1.0,
            ddim_steps=20,         # lass_inference 기본과 동일
            n_gen=1,
            save_mixed=False       # eval 중 mixed wav 저장은 끔
        )

        # waveform shape 처리
        # mel_spectrogram_to_waveform() 구현 상
        # 보통 (B, 1, T) 또는 (B, T) 형태
        if isinstance(waveform, np.ndarray):
            wav_np = waveform
        else:
            wav_np = np.array(waveform)

        if wav_np.ndim == 3:      # (B, 1, T)
            sep = wav_np[0, 0]
        elif wav_np.ndim == 2:    # (B, T)
            sep = wav_np[0]
        else:                     # 기타 케이스 방어적 처리
            sep = wav_np.reshape(-1)

        sep_tensor = torch.from_numpy(sep).float().view(1, 1, -1).to(self.device)

        return {"waveform": sep_tensor}
