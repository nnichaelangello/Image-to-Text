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
        img_np = np.array(img, dtype=np.uint8)
        clahe_img = self.clahe.apply(img_np)
        return Image.fromarray(clahe_img)


class AddCannyChannel:
    """
    Input tensor: [1, H, W] (grayscale)
    Output tensor: [2, H, W] (grayscale + canny)
    """
    def __call__(self, tensor_img):
        # ke numpy
        img_np = tensor_img.squeeze(0).numpy() * 255
        img_np = img_np.astype(np.uint8)

        # canny
        edges = cv2.Canny(img_np, threshold1=100, threshold2=200)

        # normalisasi edges (0–1)
        edges = edges.astype(np.float32) / 255.0

        # stack channel
        edges_tensor = torch.tensor(edges).unsqueeze(0)
        stacked = torch.cat([tensor_img, edges_tensor], dim=0)

        return stacked


transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    CLAHETransform(clip_limit=3.0, tile_grid_size=(8,8)),

    transforms.RandomApply([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
    ], p=0.5),

    transforms.RandomApply([
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)
    ], p=0.5),

    transforms.ToTensor(),

    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),
    transforms.Normalize((0.1307,), (0.3081,)),

    AddCannyChannel(),   # ← channel stacking
])


transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    AddCannyChannel(),   # test juga 2 channel
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
