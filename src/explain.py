#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small
from torchvision.transforms.functional import resize


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanations for representative predictions.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def select_records(path):
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    selected = []
    for class_name in sorted({record["actual"] for record in records}):
        candidates = [record for record in records if record["actual"] == class_name and record["correct"]]
        selected.append(max(candidates, key=lambda record: record["confidence"]))
    errors = sorted((record for record in records if not record["correct"]), key=lambda record: record["confidence"], reverse=True)
    selected.extend(errors[:4])
    return selected


def main():
    args = parse_args()
    model, checkpoint = load_model(args.model)
    image_size = checkpoint["image_size"]
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    activations = []
    gradients = []
    layer = model.features[-1]
    forward_hook = layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    backward_hook = layer.register_full_backward_hook(lambda _module, _grad_input, grad_output: gradients.append(grad_output[0]))

    records = select_records(args.predictions)
    fig, axes = plt.subplots(2, 4, figsize=(14, 8.5))
    for axis, record in zip(axes.flat, records):
        with Image.open(record["image_path"]) as source:
            source = source.convert("RGB")
            display = transforms.Compose([
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
            ])(source)
            tensor = transform(source).unsqueeze(0)
        activations.clear()
        gradients.clear()
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        predicted_id = logits.argmax(1).item()
        logits[0, predicted_id].backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        cam -= cam.min()
        cam /= cam.max().clamp_min(1e-8)
        cam = resize(cam, [image_size, image_size], antialias=True).squeeze().detach().numpy()
        axis.imshow(display)
        axis.imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
        status = "Correct" if record["correct"] else "Error"
        axis.set_title(
            f"{status}: {record['predicted'].replace('_', ' ').title()}\n"
            f"True: {record['actual'].replace('_', ' ').title()} | {record['confidence']:.1%}",
            fontsize=9,
        )
        axis.axis("off")
    forward_hook.remove()
    backward_hook.remove()
    fig.suptitle("Grad-CAM: image regions influencing representative predictions", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.90, wspace=0.12, hspace=0.34)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
