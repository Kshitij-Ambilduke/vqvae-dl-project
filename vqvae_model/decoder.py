import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, embedding_dim, channels=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(embedding_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.layers(x)