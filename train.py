import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from model.shared_cnn_backbone import SharedCNNBackbone
from model.itt_head import ITTHead
from dataset import get_loaders
from config import Config as cfg

def main():
    device = torch.device(cfg.device)
    backbone = SharedCNNBackbone().to(device)
    head = ITTHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(list(backbone.parameters()) + list(head.parameters()), lr=cfg.lr)
    train_loader, test_loader = get_loaders()

    for epoch in range(1, cfg.epochs + 1):
        backbone.train()
        head.train()
        total_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}"):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            feats = backbone(images)
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        acc = evaluate(backbone, head, test_loader, device)
        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Test Accuracy: {acc*100:05.2f}%")

    torch.save(backbone.state_dict(), "shared_cnn_backbone.pth")
    torch.save(head.state_dict(), "itt_head.pth")

def evaluate(backbone, head, loader, device):
    backbone.eval()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            feats = backbone(images)
            preds = head(feats).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

if __name__ == "__main__":
    main()