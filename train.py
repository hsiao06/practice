import torch
import torch.nn as nn
import glob
import os

import matplotlib.pyplot as plt

from utils import extract_log_mel_features, kmeans
from run_speech_and_music_kmeans import wav_file, target_sampling_rate, n_fft, n_mels, n_clusters

# Setting
hidden_dim = 32
out_channels = 32
time_kernel = 5
pool_kernel = 21  # smoothing window (frames) over time, ~210ms at a 10ms frame shift
num_epochs = 200
learning_rate = 1e-2
seed = 0


def load_audio(audio_dir):
    wavforms = []
    sample_rates = []
    for audio_file in glob.glob(os.path.join(audio_dir, "*.wav")):
        # wavform, sample_rate = librosa.load(audio_file)
        waveform, sample_rate = torchaudio.load(audio_file)
        wavforms.append(wavform)
        sample_rates.append(sample_rate)
    return wavforms, sample_rates


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self, n_mels, out_channels, time_kernel, pool_kernel, n_classes):
        super().__init__()
        # 2D filter: spans the whole frequency band (n_mels) and `time_kernel` frames
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=(n_mels, time_kernel),
            padding=(0, time_kernel // 2),
        )
        self.relu = nn.ReLU()
        # smooth the per-frame features over time so predictions don't flicker frame-to-frame
        self.pool = nn.AvgPool1d(
            kernel_size=pool_kernel,
            stride=1,
            padding=pool_kernel // 2,
            count_include_pad=False,
        )
        self.linear = nn.Linear(out_channels, n_classes)

    def forward(self, x):
        # x: (n_mels, n_frames)
        x = x.unsqueeze(0).unsqueeze(0)     # (1, 1, n_mels, n_frames)
        x = self.relu(self.conv(x))         # (1, out_channels, 1, n_frames)
        x = x.squeeze(2)                    # (1, out_channels, n_frames)
        x = self.pool(x)                    # (1, out_channels, n_frames)
        x = x.squeeze(0).transpose(0, 1)    # (n_frames, out_channels)
        return self.linear(x)               # (n_frames, n_classes)

def main():
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    #waveforms, sampling_rates = load_audio("./wav")

    features = list()

    for audio_file in glob.glob(os.path.join("./wav", "*.wav")):
        # Reuse the same log-mel features and k-means clusters as run_speech_and_music_kmeans.py
        log_mel_spec, _ = extract_log_mel_features(wav_file, target_sampling_rate, n_fft, n_mels)
        log_mel_spec = log_mel_spec.transpose(0, 1)  # (n_frames, n_mels)
        features.append(log_mel_spec)

    model = MLP(n_mels, hidden_dim, n_mels)
    #model = CNN(n_mels, out_channels, time_kernel, pool_kernel, n_mels)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    #print(features[0][0:10])

    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss_epoch = 0.0

        for x in features:
            logits = model(x)
            loss = criterion(logits, x)
            loss_epoch += loss.item()
            loss.backward()
        
        optimizer.step()
        print(epoch, loss_epoch)
        #losses.append(loss.item())

#        if (epoch + 1) % 20 == 0 or epoch == 0:
#            with torch.no_grad():
#                preds = logits.argmax(dim=1)
#                accuracy = (preds == labels).float().mean().item()
#            print(f"epoch {epoch + 1:4d}/{num_epochs}  loss {loss.item():.4f}  accuracy {accuracy:.4f}")

    # make the plots: training loss curve and predicted vs k-means cluster assignment
#    with torch.no_grad():
#        preds = model(features).argmax(dim=1).numpy()
#    labels_np = labels.numpy()
#
#    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
#    axs[0].plot(losses)
#    axs[0].set_title("Training loss")
#    axs[0].set_xlabel("Epoch")
#    axs[0].set_ylabel("Cross-entropy loss")
#    axs[0].grid(True)
#
#    axs[1].imshow(labels_np[None, :], aspect="auto", cmap="tab10", vmin=0, vmax=n_clusters - 1)
#    axs[1].set_yticks([])
#    axs[1].set_title("K-means cluster assignment (target)")
#
#    axs[2].imshow(preds[None, :], aspect="auto", cmap="tab10", vmin=0, vmax=n_clusters - 1)
#    axs[2].set_yticks([])
#    axs[2].set_xlabel("Frame")
#    axs[2].set_title("MLP prediction")
#
#    plt.tight_layout()
#    plt.show()


if __name__ == "__main__":
    main()
