import sys
import time

import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import librosa
import matplotlib.pyplot as plt
from plot import plot_waveform, plot_spectrogram, plot_fbank

# Setting
wav_file = "wav/epi.wav"
target_sampling_rate = 8000
n_fft = 512
n_mels = 80
eps = 1e-10

def main():
    # load audio and resample to target sampling rate
    waveform, sampling_rate = torchaudio.load(wav_file)
    waveform = torchaudio.functional.resample(waveform, sampling_rate, target_sampling_rate)
    sampling_rate = target_sampling_rate

    window_length = int(0.025 * sampling_rate) # 25ms window
    window_shift = sampling_rate // 100        # 1/100th of a second == 10ms

    # Create spectrogram
    spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=window_shift,  win_length=window_length)
    spec = spectrogram(waveform)
    
    mel_filters = F.melscale_fbanks(
        int(n_fft // 2 + 1),
        n_mels=n_mels,
        f_min=0.0,
        f_max=sampling_rate / 2.0,
        sample_rate=sampling_rate,
        mel_scale="htk",
        norm=None,
    )
    
    # Compute mel and log mel fbank features
    mel_spec = torch.matmul(spec.transpose(-1, -2), mel_filters).transpose(-1, -2)
    log_mel_spec = torch.log(mel_spec + eps)
   
    # make the plots
    fig, axs = plt.subplots(4, 1)
    plot_waveform(waveform, sampling_rate, title="Original waveform", ax=axs[0])
    plot_spectrogram(spec[0], title="spectrogram", ax=axs[1], sr=sampling_rate, n_fft=n_fft)
    plot_spectrogram(mel_spec[0], title="mel spectrogram", ax=axs[2])
    plot_spectrogram(log_mel_spec[0], title="log mel spectrogram", ax=axs[3])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
