import torch
import torch.nn as nn
from torch.quantization import quantize_fx

from copy import deepcopy

from .mnist_mlp import train_with_early_stopping, test_model

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

class BANKNOTE_MLP(nn.Module):  # 36 pesi
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


def load_banknote_dataset(batch_size=32):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    column_names = ["variance", "skewness", "curtosis", "entropy", "class"]
    df = pd.read_csv(url, header=None, names=column_names)

    X = df.drop("class", axis=1).values
    y = df["class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, train_size=0.7, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def main():
    root = "../../../../data"
    
    train_loader, validation_loader, test_loader = load_banknote_dataset()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BANKNOTE_MLP()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    model, history = train_with_early_stopping(model, train_loader, validation_loader,
                            optimizer, criterion, device, save_path="best_banknote_mlp.pt")
    
    test_model(model, test_loader, device, load_path="./weights/best_banknote_mlp.pt")
    
    ## FX GRAPH
    backend = "fbgemm" 
    model.cpu()
    model.eval()
    m = deepcopy(model)
    m.eval()
    qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
    # Prepare
    calibration_loader = DataLoader(Subset(train_loader.dataset, range(100)), batch_size=32, shuffle=False)
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

def get_banknote_mlp(train_loader, test_loader, load_path="./weights/best_banknote_mlp.pt", quantized=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BANKNOTE_MLP().to(device)
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