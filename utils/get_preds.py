import numpy as np
import torch

def get_preds(test_loader, model, device, verb=False):
    correct = 0
    total = 0
    tot_p = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            # classificazione standard
            _, preds = torch.max(outputs, dim=1)
            tot_p.extend(preds.cpu().numpy())
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    test_accuracy = correct / total
    if verb:
        print(f"🧪 Test Accuracy: {test_accuracy:.4f}")
    return test_accuracy, np.array(tot_p)