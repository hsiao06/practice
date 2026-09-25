import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


def kmeans(x, n_clusters, n_iters=100, tol=1e-4, generator=None):
    """Lloyd's algorithm k-means. x: (n_samples, n_features) -> (labels, centroids)."""
    n_samples = x.shape[0]
    init_idx = torch.randperm(n_samples, generator=generator)[:n_clusters]
    centroids = x[init_idx].clone()

    for _ in range(n_iters):
        dists = torch.cdist(x, centroids)  # (n_samples, n_clusters)
        labels = dists.argmin(dim=1)

        new_centroids = centroids.clone()
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centroids[k] = x[mask].mean(dim=0)

        shift = torch.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids


def extract_log_mel_features(wav_file, target_sampling_rate, n_fft, n_mels, eps=1e-10):
    """Load wav_file and return (log_mel_spec_avg, sampling_rate), log_mel_spec_avg: (n_mels, n_frames)."""
    # load audio and resample to target sampling rate
    waveform, sampling_rate = torchaudio.load(wav_file)
    waveform = torchaudio.functional.resample(waveform, sampling_rate, target_sampling_rate)
    sampling_rate = target_sampling_rate

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

    # Compute mel and log mel fbank features (batched over channels), then average over channels
    mel_spec = torch.matmul(spec.transpose(-1, -2), mel_filters).transpose(-1, -2)
    log_mel_spec = torch.log(mel_spec + eps)
    log_mel_spec_avg = log_mel_spec.mean(dim=0)  # (n_mels, n_frames)

    return log_mel_spec_avg, sampling_rate


