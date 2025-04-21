from pixelcnn.masked_cnn import MaskedConv2d
import torch.nn as nn

class PixelCNN(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, 64)

        self.layers = nn.Sequential(
            MaskedConv2d('A', 64, 64, 7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, padding=3),
            nn.ReLU(),
            MaskedConv2d('B', 64, 64, 7, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, num_embeddings, 1)
        )

    def forward(self, x):
        x = self.embedding(x).permute(0, 3, 1, 2)
        return self.layers(x)