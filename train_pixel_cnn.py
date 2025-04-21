import torch
import torch.nn as nn
from pixelcnn.pixel_cnn_model import PixelCNN
from tqdm import tqdm
import argparse
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader_vqvae.dataloaders_vqvae import get_dataloader
from vqvae_model.vq_vae_model import VQVAE 

def load_vqvae_model(vqvae_path, args):
    if args.dataset == "mnist" or args.dataset == "fashion_mnist":
        in_channels = 1
    else:
        in_channels = 3
    vqvae = VQVAE(
        in_channels=in_channels,
        num_embeddings=512,
        embedding_dim=64,
        beta=0.25
    )
    vqvae.load_state_dict(torch.load(vqvae_path, map_location='cpu'))
    vqvae.eval() 
    return vqvae

def create_data(model, train_loader):
    device = next(model.parameters()).device
    encodings = []
    model.eval()
    with torch.no_grad():
        for x, _ in train_loader:
            x = x.to(device)
            _, _, _, indices = model(x)
            encodings.append(indices.cpu())
    encodings = torch.cat(encodings).long()
    encoding_dataset = torch.utils.data.TensorDataset(encodings, encodings)
    encoding_loader = torch.utils.data.DataLoader(
        encoding_dataset, batch_size=256, shuffle=True)
    return encoding_loader

def main():
    parser = argparse.ArgumentParser(description="Train PixelCNN")
    parser.add_argument("--vqvae_path", type=str, required=True, help="Path to the VQ-VAE model")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for the trained model")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--num_embeddings", type=int, default=512, help="Number of embeddings")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of embeddings")
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for VQ-VAE loss")
    parser.add_argument("--in_channels", type=int, default=1, help="Number of input channels")

    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # loading original data
    train_loader,_ = get_dataloader(dataset_name=args.dataset, batch_size=64)

    # loading the models
    vqvae = load_vqvae_model(args.vqvae_path, args).to(device)
    pixelcnn = PixelCNN(num_embeddings=512).to(device)
    optimizer_pcnn = torch.optim.Adam(pixelcnn.parameters(), lr=1e-3)

    #creating pixelcnn dataste
    encoding_loader = create_data(vqvae, train_loader)

    for epoch in range(30):
        total_loss = 0
        for x, targets in tqdm(encoding_loader, desc=f"PixelCNN Epoch {epoch+1}"):
            x, targets = x.to(device), targets.to(device)
            optimizer_pcnn.zero_grad()

            logits = pixelcnn(x)
            loss = F.cross_entropy(logits, targets)

            loss.backward()
            optimizer_pcnn.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss/len(encoding_loader):.4f}")
    torch.save(pixelcnn.state_dict(), f"{args.output_dir}/pixelcnn_{epoch+1}.pth")

if __name__ == "__main__":
    main()



