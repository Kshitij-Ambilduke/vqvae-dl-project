from vqvae_model.encoder import Encoder
from vqvae_model.decoder import Decoder 
from vqvae_model.vector_quantizer import VectorQuantizer
import torch.nn as nn

class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, beta=0.25, in_channels=3):
        super().__init__()
        self.encoder = Encoder(embedding_dim, in_channels)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, beta)
        self.decoder = Decoder(channels=in_channels, embedding_dim=embedding_dim)

    def forward(self, x):
        z = self.encoder(x)
        quantized, codebook_loss, commitment_loss, indices = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, codebook_loss, commitment_loss, indices

