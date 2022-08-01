import torch
import torchvision

cifar10_data = torchvision.datasets.CIFAR10('/share/lucuslu/ntga/chlu/datasets', download=True)
data_loader = torch.utils.data.DataLoader(cifar10_data,
                                          batch_size=16,
                                          shuffle=True,
                                          drop_last=True,
                                          num_workers=2)

print(cifar10_data)