import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model.shared_cnn_backbone import SharedCNNBackbone
from model.itt_head import ITTHead
from config import Config as cfg

preprocess = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict(image_path):
    device = torch.device(cfg.device)
    backbone = SharedCNNBackbone().to(device)
    head = ITTHead().to(device)
    backbone.load_state_dict(torch.load("shared_cnn_backbone.pth", map_location=device))
    head.load_state_dict(torch.load("itt_head.pth", map_location=device))
    backbone.eval()
    head.eval()

    img = Image.open(image_path).convert("L")
    tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = backbone(tensor)
    logit = head(feat)
    pred = logit.argmax(dim=1).item()

    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted digit: {pred}")
    plt.axis("off")
    plt.show()
    return pred

# contoh
# predict("sample_digit.png")