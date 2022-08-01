from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor

def _load(dataset_fn):
    train_data = dataset_fn(root='/share/lucuslu/ntga/chlu/datasets', train=True, download=True, transform=ToTensor())
    print("===train data===")
    print(train_data)

    test_data = dataset_fn('/share/lucuslu/ntga/chlu/datasets', train=False, download=True, transform=ToTensor())
    print("===test data===")
    print(test_data)

    return train_data, test_data

def load_datasets(dataset_name: str = 'cifar10', batch_size: int = 64, num_workers: int = 16):
    """
    :param dataset_name: string. `cifar10`, `imagenet` or `mnist`.
    :param batch_size: int.
    :param num_workers: int. Number of threads for dataset loader.
    :return: triple of callable functions (train_loader, test_loader, eps).
            In Neural Tangents, a network is defined by a triple of functions (init_fn, apply_fn, kernel_fn). 
            train_loader: train_loader for this dataset (torchvision).
            test_loader: test_loader for this dataset (torchvision).
            eps: epsilon. Strength of NTGA
    """
    print("Loading dataset...")
    if dataset_name == 'cifar10':
        train_data = _load(datasets.CIFAR10)
        eps = 8/255
    elif dataset_name == 'imagenet':
        train_data = _load(datasets.ImageNet)
        eps = 0.1
    else dataset_name == 'minst':
        train_data = _load(datasets.MNIST)
        eps = 0.3

    eps_iter = (eps/nb_iter)*1.1

    train_loader = DataLoader(train_data,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=num_workers)
    test_loader = DataLoader(test_data,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=num_workers)
    print("Fin Loading dataset (split into train, test)")
    return train_loader, test_loader, eps, eps_iter
