from .cifar import get_cifar10
from .digits_pca import load_digits_pca
from .mnist import get_mnist
from .banknote import get_banknote

datasets = {
    "cifar10": get_cifar10,
    'digits-pca': load_digits_pca,
    'mnist': get_mnist,
    'banknote': get_banknote
}

def get_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
