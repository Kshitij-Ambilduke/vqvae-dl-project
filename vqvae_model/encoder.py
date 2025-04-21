from torch import nn

class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, in_channels=3):
        super().__init__()
        
        # Input: (in_channels, 28, 28)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), # Output: (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # Output: (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, embedding_dim, 1)    # Output: (embedding_dim, 7, 7)
        )

    def forward(self, x):
        return self.layers(x)