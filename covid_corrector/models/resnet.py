"""Frozen ResNet50 loader."""

from pathlib import Path


def load_resnet(checkpoint_path: Path, device: str = "cpu"):
    import torch
    from torch import nn
    from torchvision.models import resnet50

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    return model

