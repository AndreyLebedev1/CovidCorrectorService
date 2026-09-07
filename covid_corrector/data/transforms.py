"""Inference image preprocessing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformSpec:
    input_size: int = 224
    channels: int = 3
    normalization: str = "imagenet"


def build_transforms(spec: TransformSpec):
    """Build the deterministic transform used by the frozen model."""
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("torchvision is required") from exc

    operations = [transforms.Resize((spec.input_size, spec.input_size))]
    if spec.channels == 1:
        operations.append(transforms.Grayscale(num_output_channels=1))
    else:
        operations.append(transforms.Lambda(lambda image: image.convert("RGB")))
    operations.append(transforms.ToTensor())

    if spec.normalization == "imagenet":
        operations.append(
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            )
        )
    elif spec.normalization != "unit":
        raise ValueError(f"Unknown normalization: {spec.normalization}")

    return transforms.Compose(operations)

