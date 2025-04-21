from dataloader_vqvae.dataloaders_vqvae import get_dataloader
import argparse
from vqvae_model.vq_vae_model import VQVAE
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser(description="Train VQVAE model")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use for training')
    parser.add_argument('--embedding_dim', type=int, default=64, help='Dimension of the embedding space')
    parser.add_argument('--num_embeddings', type=int, default=512, help='Number of embeddings')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment cost for VQ-VAE')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train the model')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save the model and results')
    
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'reconstruction_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'newly_sample_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'loss_plots'), exist_ok=True)

    train_loader, test_loader = get_dataloader(
        batch_size=args.batch_size, dataset_name=args.dataset
    )


    if args.dataset == 'mnist' or args.dataset == 'fashion_mnist':
        in_channels = 1
    else:
        in_channels = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the VQVAE model
    model  = VQVAE(
        in_channels=in_channels,
        num_embeddings=args.num_embeddings,
        embedding_dim=args.embedding_dim,
        beta=args.beta
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_recon, train_codebook, train_commit, train_total = [], [], [], []
    val_recon, val_codebook, val_commit, val_total = [], [], [], []

    # Fixed test batch for reconstruction visualization
    test_iter = iter(test_loader)
    fixed_test, _ = next(test_iter)
    fixed_test = fixed_test.to(device)

    # Training loop
    for epoch in range(args.num_epochs):
        model.train()
        epoch_recon, epoch_codebook, epoch_commit, epoch_total = 0, 0, 0, 0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x = x.to(device)
            optimizer.zero_grad()

            x_recon, codebook_loss, commit_loss, _ = model(x)
            recon_loss = F.mse_loss(x_recon, x)
            commit_loss_scaled = model.vq.beta * commit_loss
            total_loss = recon_loss + codebook_loss + commit_loss_scaled

            total_loss.backward()
            optimizer.step()

            epoch_recon += recon_loss.item()
            epoch_codebook += codebook_loss.item()
            epoch_commit += commit_loss_scaled.item()
            epoch_total += total_loss.item()

        # Store training losses
        train_recon.append(epoch_recon/len(train_loader))
        train_codebook.append(epoch_codebook/len(train_loader))
        train_commit.append(epoch_commit/len(train_loader))
        train_total.append(epoch_total/len(train_loader))

        # Validation
        model.eval()
        v_recon, v_codebook, v_commit, v_total = 0, 0, 0, 0
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(device)
                x_recon, codebook_loss, commit_loss, _ = model(x)
                recon_loss = F.mse_loss(x_recon, x)
                commit_loss_scaled = model.vq.beta * commit_loss
                total_loss = recon_loss + codebook_loss + commit_loss_scaled

                v_recon += recon_loss.item()
                v_codebook += codebook_loss.item()
                v_commit += commit_loss_scaled.item()
                v_total += total_loss.item()

        # Store validation losses
        val_recon.append(v_recon/len(test_loader))
        val_codebook.append(v_codebook/len(test_loader))
        val_commit.append(v_commit/len(test_loader))
        val_total.append(v_total/len(test_loader))

        # Plot reconstructions
        with torch.no_grad():
            recon_images, _, _, _ = model(fixed_test)
    
        fig, axes = plt.subplots(2, 10, figsize=(20, 4))
        for i in range(10):
            axes[0, i].imshow(fixed_test[i].cpu().squeeze().numpy() * 0.5 + 0.5, cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon_images[i].cpu().squeeze().numpy() * 0.5 + 0.5, cmap='gray')
            axes[1, i].axis('off')

        plt.suptitle(f'Epoch {epoch+1} Reconstructions')
        # Save instead of showing
        plt.savefig(f'{args.output_dir}/reconstruction_images/recon_epoch_{epoch+1}.png', bbox_inches='tight', dpi=150)
        plt.close()
    
    #save the model
    # torch.save(model, os.path.join(args.output_dir, f'vqvae_model_{epoch+1}.pth'))
    torch.save(model.state_dict(), os.path.join(args.output_dir, f'vqvae_model_{epoch+1}.pth'))    
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_recon, label='Train')
    plt.plot(val_recon, label='Validation')
    plt.title('Reconstruction Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(train_codebook, label='Train')
    plt.plot(val_codebook, label='Validation')
    plt.title('Codebook Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(train_commit, label='Train')
    plt.plot(val_commit, label='Validation')
    plt.title('Commitment Loss (Scaled)')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(train_total, label='Train')
    plt.plot(val_total, label='Validation')
    plt.title('Total Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/loss_plots/losses.png', bbox_inches='tight', dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
    
    

