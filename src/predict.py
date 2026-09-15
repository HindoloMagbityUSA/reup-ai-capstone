#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


def parse_args():
    parser = argparse.ArgumentParser(description="Predict a ReUP listing's material type.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("model/best_model.pt"))
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = torch.load(args.model, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint["image_size"]
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with Image.open(args.image) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor) / float(checkpoint.get("temperature", 1.0))
        probs = torch.softmax(logits, dim=1).squeeze(0)
    confidence, prediction = probs.max(0)
    result = {
        "material_suggestion": class_names[prediction.item()],
        "confidence": confidence.item(),
        "probabilities": {name: probability.item() for name, probability in zip(class_names, probs)},
        "verification_required": True,
        "model_scope": "material_type_only",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

