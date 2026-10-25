import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class CoffeeDiseaseClassifier(nn.Module):
    """
    ResNet18-based Classifier for Coffee Leaf Disease Detection.
    Matches the exact training architecture and checkpoint weight layers.
    """
    def __init__(self, num_classes: int = 4, freeze_features: bool = False):
        super(CoffeeDiseaseClassifier, self).__init__()
        # Use weights=None to avoid deprecation warnings while keeping pretrained=False behavior
        try:
            self.backbone = models.resnet18(weights=None)
        except TypeError:
            self.backbone = models.resnet18(pretrained=False)

        # Freeze early layers if training mode was configured
        if freeze_features:
            for param in list(self.backbone.parameters())[:-20]:
                param.requires_grad = False

        # Replace final classification head (exact structure from training)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features=in_features, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
