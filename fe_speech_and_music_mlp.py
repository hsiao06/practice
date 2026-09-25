import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from fe_speech_and_music_kmeans import extract_log_mel_features, kmeans, n_mels, n_clusters

# Setting
hidden_dim = 32
num_epochs = 200
learning_rate = 1e-2
seed = 0


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


def main():
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)

    # Reuse the same log-mel features and k-means clusters as fe_speech_and_music_kmeans.py
    log_mel_spec_avg, _ = extract_log_mel_features()
    features = log_mel_spec_avg.transpose(0, 1)  # (n_frames, n_mels)
    labels, _ = kmeans(features, n_clusters, generator=generator)  # (n_frames,), used as training targets

    model = MLP(n_mels, hidden_dim, n_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (epoch + 1) % 20 == 0 or epoch == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                accuracy = (preds == labels).float().mean().item()
            print(f"epoch {epoch + 1:4d}/{num_epochs}  loss {loss.item():.4f}  accuracy {accuracy:.4f}")

    # make the plots: training loss curve and predicted vs k-means cluster assignment
    with torch.no_grad():
        preds = model(features).argmax(dim=1).numpy()
    labels_np = labels.numpy()

    fig, axs = plt.subplots(3, 1, figsize=(10, 7))
    axs[0].plot(losses)
    axs[0].set_title("Training loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross-entropy loss")
    axs[0].grid(True)

    axs[1].imshow(labels_np[None, :], aspect="auto", cmap="tab10", vmin=0, vmax=n_clusters - 1)
    axs[1].set_yticks([])
    axs[1].set_title("K-means cluster assignment (target)")

    axs[2].imshow(preds[None, :], aspect="auto", cmap="tab10", vmin=0, vmax=n_clusters - 1)
    axs[2].set_yticks([])
    axs[2].set_xlabel("Frame")
    axs[2].set_title("MLP prediction")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
