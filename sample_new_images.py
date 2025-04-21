import torch
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
from vqvae_model.vq_vae_model import VQVAE
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'myfolder')))
from pixelcnn.pixel_cnn_model import PixelCNN


def load_pixelcnn_model(model_path):
    model = PixelCNN(num_embeddings=512)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))#) = torch.load(model_path, map_location='cpu').to(device)
    model.eval()
    return model

def load_vqvae_model(vqvae_path):
    vqvae = VQVAE(
        in_channels=1,
        num_embeddings=512,
        embedding_dim=64,
        beta=0.25
    )
    vqvae.load_state_dict(torch.load(vqvae_path, map_location='cpu'))
    vqvae.eval() 
    return vqvae

def sample_pixelcnn(model, batch_size, device='cpu'):
    model.eval()
    with torch.no_grad():
        # print(batch_size, device)
        samples = torch.zeros((batch_size, 7, 7)).long()
        samples = samples.to(device)
        for i in range(7):
            for j in range(7):
                logits = model(samples)
                probs = F.softmax(logits[:, :, i, j], dim=1)
                samples[:, i, j] = torch.multinomial(probs, 1).squeeze()
        return samples

def main():
    parser = argparse.ArgumentParser(description='Sample images from PixelCNN and VQVAE models')
    parser.add_argument('--pixelcnn_path', type=str, required=True, help='Path to the PixelCNN model')
    parser.add_argument('--vqvae_path', type=str, required=True, help='Path to the VQVAE model')
    parser.add_argument('--output_dir', type=str, required=True, help='Output dir')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    pixelcnn = load_pixelcnn_model(args.pixelcnn_path).to(device)
    vqvae = load_vqvae_model(args.vqvae_path).to(device)

    # Generate and decode samples
    for time in range(5):
        samples = sample_pixelcnn(pixelcnn, batch_size=10, device=device)   
        quantized = vqvae.vq.embedding(samples).permute(0, 3, 1, 2)
        with torch.no_grad():
            generated = vqvae.decoder(quantized).cpu()
    # Plot generated images
        fig, axes = plt.subplots(1, 10, figsize=(20, 2))
        for i in range(10):
            axes[i].imshow(generated[i].squeeze().numpy()*0.5+0.5, cmap='gray')
            axes[i].axis('off')
        plt.savefig(f"{args.output_dir}/newly_sample_images/generated_samples_{time+1}.png")
    
if __name__ == '__main__':
    main()