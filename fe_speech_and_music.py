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
wav_file = "wav/speech-and-music.wav"
target_sampling_rate = 16000
n_fft = 512
n_mels = 80
eps = 1e-10

def main():
    # load audio and resample to target sampling rate
    waveform, sampling_rate = torchaudio.load(wav_file)
    waveform = torchaudio.functional.resample(waveform, sampling_rate, target_sampling_rate)
    sampling_rate = target_sampling_rate
   
    #waveform = waveform.mean(dim=0, keepdim=True)
    num_channels = waveform.shape[0]

    window_length = int(0.025 * sampling_rate) # 25ms window
    window_shift = sampling_rate // 100        # 1/100th of a second == 10ms

    # Create spectrogram (batched over channels)
    spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=window_shift, win_length=window_length)
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

    # Compute mel and log mel fbank features (batched over channels)
    mel_spec = torch.matmul(spec.transpose(-1, -2), mel_filters).transpose(-1, -2)
    log_mel_spec = torch.log(mel_spec + eps)
    log_mel_spec_avg = log_mel_spec.mean(dim=0, keepdim=True)

    # make the plots: one column per channel, one row per feature stage
    fig, axs = plt.subplots(5, num_channels, figsize=(6 * num_channels, 10))
    for ch in range(num_channels):
        plot_waveform(waveform, sampling_rate, title=f"Channel {ch} waveform", ax=axs[0, ch], channel=ch)
        plot_spectrogram(spec[ch], title=f"Channel {ch} spectrogram", ax=axs[1, ch], sr=sampling_rate, n_fft=n_fft)
        plot_spectrogram(mel_spec[ch], title=f"Channel {ch} mel spectrogram", ax=axs[2, ch])
        plot_spectrogram(log_mel_spec[ch], title=f"Channel {ch} log mel spectrogram", ax=axs[3, ch])
        plot_spectrogram(log_mel_spec_avg[0], title=f"Avg log mel spectrogram", ax=axs[4, ch])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
