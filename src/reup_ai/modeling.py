from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


def build_classifier(num_classes: int):
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def load_classifier(checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = build_classifier(len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def inference_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def predict_classifier(image_path: Path, checkpoint_path: Path) -> dict:
    model, checkpoint = load_classifier(checkpoint_path)
    transform = inference_transform(int(checkpoint["image_size"]))
    with Image.open(image_path) as image:
        tensor = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        temperature = float(checkpoint.get("temperature", 1.0))
        probabilities = torch.softmax(model(tensor) / temperature, dim=1).squeeze(0)
    confidence, prediction = probabilities.max(0)
    class_names = checkpoint["class_names"]
    return {
        "task": checkpoint.get("task", "unknown"),
        "suggestion": class_names[prediction.item()],
        "confidence": float(confidence.item()),
        "probabilities": {
            name: float(probability.item())
            for name, probability in zip(class_names, probabilities)
        },
    }
