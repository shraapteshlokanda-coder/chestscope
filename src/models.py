import torch.nn as nn
from torchvision import models


class SimpleCNN(nn.Module):
    """Baseline: 4 conv blocks trained from scratch."""

    def __init__(self):
        super().__init__()

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                nn.ReLU(), nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.head(self.features(x))


def build_resnet18(freeze_backbone=True, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=weights)
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
    m.fc = nn.Linear(m.fc.in_features, 1)  # new head, always trainable
    return m


def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def unfreeze_layers(model, names=("layer3", "layer4", "fc")):
    """Unfreeze only the named top-level modules (e.g. layer3, layer4, fc)."""
    for n, p in model.named_parameters():
        if n.split(".")[0] in names:
            p.requires_grad = True