from pipeline import build_audiosep, separate_audio
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_audiosep(
    config_yaml="config/audiosep_base.yaml",
    checkpoint_path="checkpoint/audiosep_base_4M_steps.ckpt",
    device=device,
)

audio_file = "input/lullaby_short.mp3"
text = "female vocal only"
output_file = "output/lullaby_short_female_vocal_only_COMP.wav"

text = "Lullaby"
output_file = "output/lullaby_short_Lullaby_AUDIOSEP.wav"

# ----------------------------------------------
# AudioSep processes the audio at 32 kHz sampling rate
separate_audio(model, audio_file, text, output_file, device)
