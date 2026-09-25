import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

import matplotlib.pyplot as plt
from plot import plot_spectrogram
from utils import kmeans, extract_log_mel_features

# Setting
wav_file = "wav/speech-and-music.wav"
target_sampling_rate = 16000
n_fft = 512
n_mels = 80
eps = 1e-10
n_clusters = 2
kmeans_iters = 100
kmeans_tol = 1e-4


def main():
    log_mel_spec_avg, sampling_rate = extract_log_mel_features(wav_file, target_sampling_rate, n_fft, n_mels)

    # Each time frame's log-mel vector (n_mels dims) is one sample to cluster
    features = log_mel_spec_avg.transpose(0, 1)  # (n_frames, n_mels)
    generator = torch.Generator().manual_seed(0)
    labels, _ = kmeans(features, n_clusters, n_iters=kmeans_iters, tol=kmeans_tol, generator=generator)
    labels = labels.numpy()  # (n_frames,)

    # make the plots: log mel spectrogram with the per-frame cluster assignment below it
    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [4, 1]})
    plot_spectrogram(log_mel_spec_avg, title="Avg log mel spectrogram", ax=axs[0])
    axs[1].imshow(labels[None, :], aspect="auto", cmap="tab10", vmin=0, vmax=n_clusters - 1)
    axs[1].set_yticks([])
    axs[1].set_xlabel("Frame")
    axs[1].set_title(f"K-means cluster assignment (k={n_clusters})")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
