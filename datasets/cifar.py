import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_cifar10(root = "data",
                num_workers=8,
                train_batch_size=128,
                eval_batch_size=256):
    
    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=num_workers)
    calibration_loader = DataLoader(Subset(train_dataset, range(256)), batch_size=128, shuffle=False)

    return train_loader, val_loader, calibration_loader
