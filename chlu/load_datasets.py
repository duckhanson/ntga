from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_datasets(name: str = 'cifar10', batch_size: int = 64, num_workers: int = 16):
    print("Loading dataset...")
    train_data = datasets.CIFAR10('/share/lucuslu/ntga/chlu/datasets', train=True, download=True, transform=ToTensor())
    print("===train data===")
    print(train_data)
    test_data = datasets.CIFAR10('/share/lucuslu/ntga/chlu/datasets', train=False, download=True, transform=ToTensor())
    print("===test data===")
    print(test_data)

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
    return train_loader, test_loader
