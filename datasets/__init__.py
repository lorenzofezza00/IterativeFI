from .cifar import get_cifar10
from .mnist import get_mnist
from .banknote import get_banknote

datasets = {
    "cifar10": get_cifar10,
    'mnist': get_mnist,
    'banknote': get_banknote
}

def get_dataloaders(name, **kwargs):
    """Dataloaders"""
    return datasets[name.lower()](**kwargs)
