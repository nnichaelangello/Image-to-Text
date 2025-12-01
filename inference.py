import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model.shared_cnn_backbone import SharedCNNBackbone
from model.itt_head import ITTHead
from config import Config as cfg
import cv2
import numpy as np

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def apply_clahe(pil_img):
    """Terapkan CLAHE pada citra grayscale."""
    img_np = np.array(pil_img)

    if len(img_np.shape) == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(img_np)

    return Image.fromarray(cl_img)

def predict(image_path):
    device = torch.device(cfg.device)
    backbone = SharedCNNBackbone().to(device)
    head = ITTHead().to(device)
    backbone.load_state_dict(torch.load("shared_cnn_backbone.pth", map_location=device))
    head.load_state_dict(torch.load("itt_head.pth", map_location=device))
    backbone.eval()
    head.eval()

    # 1. Baca gambar
    img = Image.open(image_path).convert("L")

    # 2. Tingkatkan kontras dengan CLAHE
    img_clahe = apply_clahe(img)

    # 3. Preprocessing normal
    tensor = preprocess(img_clahe).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = backbone(tensor)
        logit = head(feat)
        pred = logit.argmax(dim=1).item()

    plt.figure(figsize=(5,3))
    plt.subplot(1,2,1)
    plt.imshow(img, cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(img_clahe, cmap="gray")
    plt.title(f"CLAHE → Pred: {pred}")
    plt.axis("off")

    plt.show()

    return pred

# contoh
# predict("sample_digit.png")
