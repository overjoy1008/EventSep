import torch
import torch.nn as nn
import numpy as np
import librosa

from utilities.data.dataset import AudioDataset


class FlowSepSeparationWrapper(nn.Module):
    """
    FlowSep latent diffusion 모델을 AudioSep evaluator의 ss_model 인터페이스로 감싸는 wrapper.
    forward()는 반드시 {"waveform": tensor, "random_start": int} 를 반환한다.
    """

    def __init__(self, flowsep_model, configs, device):
        super().__init__()
        self.model = flowsep_model
        self.device = device

        # FlowSep dataset: random_start 및 mel/STFT 기능 제공
        self.val_dataset = AudioDataset(configs, split="test", add_ons=[])

        # FlowSep은 16kHz 기준으로 학습됨
        self.sr = 16000

    def forward(self, input_dict):
        """
        input_dict:
            - "mixture": (1, 1, T) @ 32kHz (AudioSep evaluator가 넣어줌)
            - "text": string (query)

        return:
            {
                "waveform": tensor(1,1,T16k)  # 16k waveform
                "random_start": int           # 16k sample 기준
            }
        """

        mixture_32k = input_dict["mixture"]      # (1,1,T)
        text = input_dict["text"]

        # ------------------------------
        # 1) 32k mixture → numpy
        # ------------------------------
        mix_np_32k = mixture_32k.squeeze().detach().cpu().numpy().astype(np.float32)

        # ------------------------------
        # 2) 32k → 16k (FlowSep 입력은 16k)
        # ------------------------------
        mix_np_16k = librosa.resample(mix_np_32k, orig_sr=32000, target_sr=16000)

        # ------------------------------
        # 3) FlowSep random segment 얻기 (핵심)
        # ------------------------------
        # FlowSep inference의 핵심: dataset.read_wav_file()가
        #  - segment(10.24초) waveform
        #  - random_start
        # 를 반환한다.
        # --- 32k mixture numpy → 16k mixture numpy ---
        tmp_path = "_flowsep_temp_input.wav"
        import soundfile as sf
        sf.write(tmp_path, mix_np_16k, 16000)

        # --- FlowSep 공식 방식: 파일 경로를 넣어야 함 ---
        seg_16k, random_start = self.val_dataset.read_wav_file(tmp_path)


        # seg_16k: shape (1, T) => (T,)
        seg_16k = seg_16k[0]  # (T,)

        # ------------------------------
        # 4) mel / stft feature 추출 (FlowSep inference와 동일)
        # ------------------------------
        mixed_mel, stft = self.val_dataset.wav_feature_extraction(seg_16k.reshape(1, -1))

        if isinstance(mixed_mel, np.ndarray):
            mixed_mel_t = torch.from_numpy(mixed_mel).float()
        else:
            mixed_mel_t = mixed_mel.float()

        if isinstance(stft, np.ndarray):
            stft_t = torch.from_numpy(stft).float()
        else:
            stft_t = stft.float()

        Tseg = seg_16k.shape[0]

        # ------------------------------
        # 5) FlowSep batch 구성 (lass_inference.py와 동일)
        # ------------------------------
        batch = {}
        batch["fname"]         = ["dummy.wav"]
        batch["text"]          = [text]
        batch["caption"]       = [text]
        batch["waveform"]      = torch.from_numpy(seg_16k.reshape(1,1,Tseg)).float()
        batch["log_mel_spec"]  = mixed_mel_t.reshape(1, mixed_mel_t.shape[0], mixed_mel_t.shape[1])
        batch["sampling_rate"] = torch.tensor([self.sr])
        batch["label_vector"]  = torch.zeros(1, 527)
        batch["stft"]          = stft_t.reshape(1, stft_t.shape[0], stft_t.shape[1])
        batch["mixed_waveform"] = torch.from_numpy(seg_16k.reshape(1,1,Tseg)).float()
        batch["mixed_mel"]      = mixed_mel_t.reshape(1, mixed_mel_t.shape[0], mixed_mel_t.shape[1])

        # ------------------------------
        # 6) FlowSep generate_sample() 수행
        # ------------------------------
        wav = self.model.generate_sample(
            [batch],
            name="eval",
            unconditional_guidance_scale=1.0,
            ddim_steps=20,
            n_gen=1,
            save_mixed=False,
            save=False
        )

        # return 형태 정리
        if isinstance(wav, np.ndarray):
            wav_np = wav
        else:
            wav_np = np.array(wav)

        if wav_np.ndim == 3:      # (B,1,T)
            sep = wav_np[0, 0]
        elif wav_np.ndim == 2:    # (B,T)
            sep = wav_np[0]
        else:
            sep = wav_np.reshape(-1)

        sep_tensor = torch.from_numpy(sep).float().view(1, 1, -1).to(self.device)

        # ------------------------------
        # 7) random_start 함께 반환
        # ------------------------------
        return {
            "waveform": sep_tensor,
            "random_start": int(random_start),  # 16k 기준
        }
