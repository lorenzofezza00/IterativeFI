import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import os

from torch.quantization import quantize_fx

from .mnist_mlp import train_with_early_stopping, test_model

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 7 * 7, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, 1)

        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)


def main():
    root = "../../../../data"
    
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MNIST_CNN()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model, history = train_with_early_stopping(model, train_loader, validation_loader,
                            optimizer, criterion, device, save_path="best_mnist_cnn.pt")
    
    test_model(model, test_loader, device, load_path="./weights/best_mnist_cnn.pt")
    
    ## FX GRAPH
    backend = "fbgemm" 
    model.cpu()
    model.eval()
    m = deepcopy(model)
    m.eval()
    qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
    # Prepare
    calibration_loader = DataLoader(Subset(train_dataset, range(100)), batch_size=64, shuffle=False)
    model_prepared = quantize_fx.prepare_fx(m, qconfig_dict, example_inputs=calibration_loader)
    # Calibrate - Use representative (validation) data.
    with torch.inference_mode():
        for inputs, _ in calibration_loader:
            model_prepared(inputs)

    # quantize
    model_quantized = quantize_fx.convert_fx(model_prepared)

    # Peso quantizzato del primo layer convoluzionale
    for name, module in model_quantized.named_modules():
        if hasattr(module, "weight"):
            w = module.weight()
            print(f"Layer: {name}, shape: {w.shape}, dtype: {w.dtype}")
            w_int = w.int_repr()
            print(f"Layer {name} int weights:\n{w_int}")

def get_mnist_cnn(train_loader, test_loader, load_path="./weights/best_mnist_cnn.pt", quantized=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNIST_CNN().to(device)
    test_model(model, test_loader, device, load_path=load_path)
    
    if quantized:
        ## FX GRAPH
        backend = "fbgemm" 
        model.cpu()
        model.eval()
        m = deepcopy(model)
        m.eval()
        qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
        # Prepare
        train_dataset = train_loader.dataset
        calibration_loader = DataLoader(Subset(train_dataset, range(100)), batch_size=64, shuffle=False)
        model_prepared = quantize_fx.prepare_fx(m, qconfig_dict, example_inputs=calibration_loader)
        # Calibrate - Use representative (validation) data.
        with torch.inference_mode():
            for inputs, _ in calibration_loader:
                model_prepared(inputs)

        # quantize
        model_quantized = quantize_fx.convert_fx(model_prepared)

        # Peso quantizzato del primo layer convoluzionale
        for name, module in model_quantized.named_modules():
            if hasattr(module, "weight"):
                w = module.weight()
                print(f"Layer: {name}, shape: {w.shape}, dtype: {w.dtype}")
                w_int = w.int_repr()
                print(f"Layer {name} int weights:\n{w_int}")
        return model_quantized
    
    return model

if __name__ == "__main__":
    main()