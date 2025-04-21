import torch
import torchvision
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import random_split

def load_mnist(batch_size=64):

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def load_fashion_mnist(batch_size=64):

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def load_celebA(batch_size=64):
    class CelebADataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.image_dir = os.path.expanduser(root_dir)
            self.transform = transform
            self.image_filenames = sorted([
                f for f in os.listdir(self.image_dir)
                if f.endswith(".jpg")
            ])

        def __len__(self):
            return len(self.image_filenames)

        def __getitem__(self, idx):
            img_path = os.path.join(self.image_dir, self.image_filenames[idx])
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image
    
    transform = transforms.Compose([
        transforms.Resize(28),
        transforms.CenterCrop(28),
        transforms.ToTensor()
    ])

    dataset = CelebADataset(root_dir="/content/data/img_align_celeba", transform=transform)

    #80-20 split dataset
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Apply VQ-VAE friendly transforms
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_cifar10(batch_size=64):
    transform = transforms.Compose([
                transforms.ToTensor(),                                    
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_dataloader(dataset_name, batch_size=64):
    if dataset_name == 'mnist':
        return load_mnist(batch_size)
    elif dataset_name == 'fashion_mnist':
        return load_fashion_mnist(batch_size)
    elif dataset_name == 'cifar10':
        return load_cifar10(batch_size)
    elif dataset_name == 'celeba':
        return load_celebA(batch_size)





