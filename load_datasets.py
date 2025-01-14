from torch.utils.data import DataLoader, random_split
from torch import flatten
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np


def _to_numpy(dataset):
    # return np.array(list(dataset))
    return dataset


def _load(dataset_fn, save_path: str):
    train_size, val_size, test_size = 1000, 200, 100

    train_val_data = dataset_fn(
        root=save_path, train=True, download=True, transform=ToTensor())
    # train_val_data = flatten(train_val_data, start_dim=1)
    print("===Origin train_val data===")
    print(train_val_data)

    train_data_size = int(len(train_val_data) * 0.8)
    valid_data_size = len(train_val_data) - train_data_size
    train_data, val_data = random_split(
        train_val_data, [train_data_size, valid_data_size])
    print("===Origin train data===")
    print(len(train_data))
    print("===Origin val data===")
    print(len(val_data))

    test_data = dataset_fn(root=save_path, train=False,
                           download=True, transform=ToTensor())
    # test_data = flatten(test_data, start_dim=1)
    print("===Origin test data===")
    print(test_data)

    # train_data, _ = random_split(train_data, [train_size, train_data_size - train_size])
    # val_data, _ = random_split(val_data, [val_size, valid_data_size - val_size])
    # test_data, _ = random_split(test_data, [test_size, int(len(test_data)) - test_size])

    return train_data, val_data, test_data


def load_datasets(dataset_name: str = 'mnist', batch_size: int = 1, num_workers: int = 16, save_path: str = '/share/lucuslu/ntga/chlu/datasets'):
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
    if dataset_name == 'mnist':
        train_data, val_data, test_data = _load(datasets.MNIST, save_path)
        num_classes = 10
        eps = 0.3
    elif dataset_name == 'imagenet':
        train_data, val_data, test_data = _load(datasets.ImageNet, save_path)
        num_classes = 2
        eps = 0.1
    elif dataset_name == 'cifar10':
        train_data, val_data, test_data = _load(datasets.CIFAR10, save_path)
        num_classes = 10
        eps = 8/255
    else:
        raise "Not support datasets"

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=num_workers)

    val_loader = DataLoader(val_data,
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

    # change into numpy
    train_dataset = _to_numpy(train_loader)
    val_dataset = _to_numpy(val_loader)
    test_dataset = _to_numpy(test_loader)

    return train_dataset, val_dataset, test_dataset, eps, num_classes
