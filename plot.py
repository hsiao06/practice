import torch
import librosa
import matplotlib.pyplot as plt


def plot_waveform(waveform, sr, title="Waveform", ax=None, channel=0):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[channel], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ax=None, sr=None, n_fft=512):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")
    if sr is not None:
        max_freq_hz = sr // 2
        tick_freqs_hz = range(0, max_freq_hz + 1, 1000)
        ax.set_yticks([f * n_fft / sr for f in tick_freqs_hz])
        ax.set_yticklabels([f"{f / 1000:.0f}" for f in tick_freqs_hz])
        ax.set_ylabel("Frequency (kHz)")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
