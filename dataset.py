import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config as cfg

import cv2
import numpy as np
from PIL import Image

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        # img adalah PIL Image (mode L / grayscale)
        img_np = np.array(img, dtype=np.uint8)
        
        # Terapkan CLAHE
        clahe_img = self.clahe.apply(img_np)
        
        # Kembali ke PIL Image
        return Image.fromarray(clahe_img)

transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),           # Pastikan grayscale
    CLAHETransform(clip_limit=3.0, tile_grid_size=(8,8)), 
    transforms.RandomApply([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ], p=0.5),
    transforms.RandomApply([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)
    ], p=0.5),
    transforms.ToTensor(),  # Sekarang ke tensor [C=1, H, W] dengan nilai 0-1
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # Re-normalize ke 0-1
    transforms.Normalize((0.1307,), (0.3081,))  # Mean & std MNIST
])

transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def get_loaders():
    train_set = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )
    test_set = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader
