from torchvision import datasets, transforms

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split

# Dataset e DataLoader
def get_mnist(root = './data', reduced=None, seed=42):
    train_ratio = 0.8
    validation_ratio = 0.2
    
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    
    dataset_size = len(train_dataset)
    train_size = int(train_ratio * dataset_size)
    validation_size = dataset_size - train_size
    
    train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    return train_loader, validation_loader, test_loader