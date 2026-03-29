import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import librosa
import matplotlib.pyplot as plt
import torch

from utilities import create_folder, get_filename
from models import *
from pytorch_utils import move_data_to_device
import config


def audio_tagging(args):
    """Inference audio tagging result of an audio clip.
    """

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')
    
    classes_num = config.classes_num
    labels = config.labels

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
    """(classes_num,)"""

    sorted_indexes = np.argsort(clipwise_output)[::-1]

    # Print audio tagging top probabilities
    for k in range(10):
        print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]], 
            clipwise_output[sorted_indexes[k]]))

    # Print embedding
    if 'embedding' in batch_output_dict.keys():
        embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels


def sound_event_detection(args):
    """Inference sound event detection result of an audio clip, 
    AND print threshold-based onset/offset intervals."""

    # Arugments & parameters
    sample_rate = args.sample_rate
    window_size = args.window_size
    hop_size = args.hop_size
    mel_bins = args.mel_bins
    fmin = args.fmin
    fmax = args.fmax
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    audio_path = args.audio_path
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    classes_num = config.classes_num
    labels = config.labels
    frames_per_second = sample_rate // hop_size

    # --- NEW ---
    threshold = 0.5

    # Paths
    fig_path = os.path.join('results', '{}.png'.format(get_filename(audio_path)))
    create_folder(os.path.dirname(fig_path))

    # Model
    Model = eval(model_type)
    model = Model(sample_rate=sample_rate, window_size=window_size, 
        hop_size=hop_size, mel_bins=mel_bins, fmin=fmin, fmax=fmax, 
        classes_num=classes_num)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    if 'cuda' in str(device):
        model.to(device)
    
    # Load audio
    (waveform, _) = librosa.core.load(audio_path, sr=sample_rate, mono=True)

    waveform = waveform[None, :]    # (1, audio_length)
    waveform = move_data_to_device(waveform, device)

    # Forward
    with torch.no_grad():
        model.eval()
        batch_output_dict = model(waveform, None)

    framewise_output = batch_output_dict['framewise_output'].data.cpu().numpy()[0]
    """(time_steps, classes_num)"""

    num_frames, num_classes = framewise_output.shape
    time_per_frame = hop_size / sample_rate

    print(f'\nSound event detection shape: {framewise_output.shape}')
    print(f'Using threshold = {threshold}\n')

    # ===============================
    # (NEW) Onset–offset detection
    # ===============================

    def detect_intervals(activity_vec):
        """activity_vec: (T,) boolean where True = active."""
        events = []
        active = False
        onset = None

        for i, v in enumerate(activity_vec):
            if v and not active:
                active = True
                onset = i * time_per_frame
            if not v and active:
                active = False
                offset = i * time_per_frame
                events.append((onset, offset))
        # 마지막 프레임이 active일 경우
        if active:
            events.append((onset, num_frames * time_per_frame))

        return events

    print("=== Detected Events (onset–offset in seconds) ===")
    for c in range(num_classes):
        activity = framewise_output[:, c] > threshold
        intervals = detect_intervals(activity)

        if len(intervals) > 0:
            print(f"{labels[c]}:")
            for (onset, offset) in intervals:
                print(f"  - {onset:.2f}s  →  {offset:.2f}s")
    print("=================================================\n")

    # # ==========================================
    # # Original PANNs visualization (unchanged)
    # # ==========================================
    # sorted_indexes = np.argsort(np.max(framewise_output, axis=0))[::-1]
    # top_k = 10  
    # top_result_mat = framewise_output[:, sorted_indexes[0 : top_k]]    

    # stft = librosa.core.stft(y=waveform[0].data.cpu().numpy(), n_fft=window_size, 
    #     hop_length=hop_size, window='hann', center=True)
    # frames_num = stft.shape[-1]

    # fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
    # axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
    # axs[0].set_ylabel('Frequency')
    # axs[0].set_title('Log spectrogram')
    # axs[1].matshow(top_result_mat.T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    # axs[1].xaxis.set_ticks(np.arange(0, frames_num, frames_per_second))
    # axs[1].xaxis.set_ticklabels(np.arange(0, frames_num / frames_per_second))
    # axs[1].yaxis.set_ticks(np.arange(0, top_k))
    # axs[1].yaxis.set_ticklabels(np.array(labels)[sorted_indexes[0 : top_k]])
    # axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    # axs[1].set_xlabel('Seconds')
    # axs[1].xaxis.set_ticks_position('bottom')

    # plt.tight_layout()
    # plt.savefig(fig_path)
    # print('Save visualization to {}'.format(fig_path))

    return framewise_output, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_at = subparsers.add_parser('audio_tagging')
    parser_at.add_argument('--sample_rate', type=int, default=32000)
    parser_at.add_argument('--window_size', type=int, default=1024)
    parser_at.add_argument('--hop_size', type=int, default=320)
    parser_at.add_argument('--mel_bins', type=int, default=64)
    parser_at.add_argument('--fmin', type=int, default=50)
    parser_at.add_argument('--fmax', type=int, default=14000) 
    parser_at.add_argument('--model_type', type=str, required=True)
    parser_at.add_argument('--checkpoint_path', type=str, required=True)
    parser_at.add_argument('--audio_path', type=str, required=True)
    parser_at.add_argument('--cuda', action='store_true', default=False)

    parser_sed = subparsers.add_parser('sound_event_detection')
    parser_sed.add_argument('--sample_rate', type=int, default=32000)
    parser_sed.add_argument('--window_size', type=int, default=1024)
    parser_sed.add_argument('--hop_size', type=int, default=320)
    parser_sed.add_argument('--mel_bins', type=int, default=64)
    parser_sed.add_argument('--fmin', type=int, default=50)
    parser_sed.add_argument('--fmax', type=int, default=14000) 
    parser_sed.add_argument('--model_type', type=str, required=True)
    parser_sed.add_argument('--checkpoint_path', type=str, required=True)
    parser_sed.add_argument('--audio_path', type=str, required=True)
    parser_sed.add_argument('--cuda', action='store_true', default=False)
    
    args = parser.parse_args()

    if args.mode == 'audio_tagging':
        audio_tagging(args)

    elif args.mode == 'sound_event_detection':
        sound_event_detection(args)

    else:
        raise Exception('Error argument!')