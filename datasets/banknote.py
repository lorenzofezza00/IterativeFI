import torch
from torch.utils.data import DataLoader, Subset, random_split, TensorDataset

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
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


def get_banknote():
    return load_banknote_dataset()