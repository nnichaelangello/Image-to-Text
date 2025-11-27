import torch
import torch.nn as nn

class SharedCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=0),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )
        ])
        self.gap = nn.AdaptiveAvgPool2d((1, None))
        self.dropout = nn.Dropout(0.4)
        self.proj = nn.Linear(512, 512)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        if x.size(1) == 1 and x.size(-1) == 28:
            x = x.unsqueeze(-1) if x.size(-2) == 28 else x
        for block in self.conv_blocks:
            x = block(x)
        x = self.gap(x)
        x = x.squeeze(2)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x = self.proj(x)
        return x