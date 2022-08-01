import torch
import torchvision


def load_datasets(name: str = 'cifar10'):
    print("Loading dataset...")
    train_data = torchvision.datasets.CIFAR10('/share/lucuslu/ntga/chlu/datasets', train=True, download=True, transform=ToTensor())
    print("===train data===")
    print(train_data)
    test_data = torchvision.datasets.CIFAR10('/share/lucuslu/ntga/chlu/datasets', train=False, download=True, transform=ToTensor())
    print("===test data===")
    print(test_data)

    # train_size = int(0.7 * len(full_dataset))
    # val_size = int(0.1 * len(full_dataset))
    # test_size = len(full_dataset) - train_size - val_size

    # train_data, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    # print("===train data===")
    # print(train_dataset)
    # print("===val data===")
    # print(val_dataset)
    # print("===test data===")
    # print(test_dataset)
    train_loader = torch.utils.data.DataLoader(train_data,
                                            batch_size=16,
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=2)
    # val_loader = torch.utils.data.DataLoader(val_dataset,
    #                                         batch_size=16,
    #                                         shuffle=False,
    #                                         drop_last=True,
    #                                         num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_data,
                                            batch_size=16,
                                            shuffle=False,
                                            drop_last=True,
                                            num_workers=2)
    print("Fin Loading dataset (split into train, test)")
    return train_loader, test_loader