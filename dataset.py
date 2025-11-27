import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config as cfg

transform_train = transforms.Compose([
    transforms.RandomApply([transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))], p=0.5),
    transforms.ToTensor(),
    transforms.RandomApply([transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)], p=0.5),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),
    transforms.Lambda(lambda x: torch.clamp(x * 2.0 - 0.5, 0.0, 1.0)),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_loaders():
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, test_loader