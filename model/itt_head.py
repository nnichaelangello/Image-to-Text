import torch.nn as nn

class ITTHead(nn.Module):
    def __init__(self, d_model=512, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        return self.classifier(x[:, 0, :])