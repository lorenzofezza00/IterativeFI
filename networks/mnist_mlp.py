import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
import os

from torch.quantization import quantize_fx


class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.relu3(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    scheduler=None,
    num_epochs=100,
    patience=10,
    save_path="best_model.pt"
):
    """
    Train PyTorch model with early stopping and save best model.

    Args:
        model (torch.nn.Module)
        train_loader (DataLoader)
        val_loader (DataLoader)
        optimizer (torch.optim.Optimizer)
        criterion (loss function)
        device (torch.device)
        num_epochs (int)
        patience (int): early stopping patience
        save_path (str): path to save best model

    Returns:
        model: trained model (best weights)
        history: dict with training and validation loss
    """
    directory = "./weights/"
    os.makedirs(directory, exist_ok=True)
        
    best_val_loss = float("inf")
    best_model_wts = deepcopy(model.state_dict())
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": []}

    model.to(device)

    for epoch in range(num_epochs):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        if scheduler is not None:
            scheduler.step()

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        # ===== EARLY STOPPING =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
            torch.save(best_model_wts, directory + save_path)
            epochs_no_improve = 0
            print("✅ Modello migliore salvato")
        else:
            epochs_no_improve += 1
            print(f"⏳ Nessun miglioramento ({epochs_no_improve}/{patience})")

        if epochs_no_improve >= patience:
            print("🛑 Early stopping attivato")
            break

    model.load_state_dict(best_model_wts)
    return model, history

def test_model(
    model,
    test_loader,
    device,
    load_path=None
):
    """
    Evaluate a PyTorch model on a test set (no loss).

    Args:
        model (torch.nn.Module)
        test_loader (DataLoader)
        device (torch.device)
        load_path (str, optional): path to saved model weights

    Returns:
        test_accuracy (float)
    """

    if load_path is not None:
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"📦 Modello caricato da {load_path}")

    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            if len(outputs) == 2:
                outputs = outputs[0]
            # classificazione standard
            _, preds = torch.max(outputs, dim=1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

    test_accuracy = correct / total

    print(f"🧪 Test Accuracy: {test_accuracy:.4f}")

    return test_accuracy


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

    model = MNIST_MLP()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model, history = train_with_early_stopping(model, train_loader, validation_loader,
                            optimizer, criterion, device, save_path="best_mnist_mlp.pt")
    
    test_model(model, test_loader, device, load_path="./weights/best_mnist_mlp.pt")
    
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

def get_mnist_mlp(train_loader, test_loader, load_path="./weights/best_mnist_mlp.pt", quantized=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNIST_MLP().to(device)
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