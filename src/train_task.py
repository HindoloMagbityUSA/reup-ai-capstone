#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from reup_ai.data import ManifestDataset, read_manifest_class_names


def parse_args():
    parser = argparse.ArgumentParser(description="Train a ReUP component or condition classifier.")
    parser.add_argument("--task", choices=("component_type", "visible_condition"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs-head", type=int, default=4)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    return parser.parse_args()


def select_device(requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transforms(image_size: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    correct = 0
    count = 0
    with torch.set_grad_enabled(training):
        for images, labels, _ in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            count += labels.size(0)
    return running_loss / count, correct / count


def collect_logits(model, loader, device):
    model.eval()
    logits, labels, paths = [], [], []
    with torch.no_grad():
        for images, targets, image_paths in loader:
            logits.append(model(images.to(device)).cpu())
            labels.append(targets.cpu())
            paths.extend(image_paths)
    return torch.cat(logits), torch.cat(labels), paths


def fit_temperature(logits, labels):
    temperature = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=80)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / temperature.clamp(0.05, 10.0), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(0.05, 10.0).item())


def expected_calibration_error(probabilities, labels, bins=10):
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    details = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidences > lower) & (confidences <= upper)
        if not mask.any():
            details.append({"lower": lower, "upper": upper, "count": 0})
            continue
        accuracy = float((predictions[mask] == labels[mask]).mean())
        confidence = float(confidences[mask].mean())
        ece += mask.mean() * abs(accuracy - confidence)
        details.append({"lower": lower, "upper": upper, "count": int(mask.sum()), "accuracy": accuracy, "confidence": confidence})
    return float(ece), details


def plot_training(history, output_dir):
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(epochs, [x["train_loss"] for x in history], marker="o", label="Train")
    axes[0].plot(epochs, [x["val_loss"] for x in history], marker="o", label="Validation")
    axes[0].set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, [x["train_accuracy"] for x in history], marker="o", label="Train")
    axes[1].plot(epochs, [x["val_accuracy"] for x in history], marker="o", label="Validation")
    axes[1].set(title="Training and validation accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion(cm, class_names, output_dir):
    side = max(6, len(class_names) * 0.7)
    fig, ax = plt.subplots(figsize=(side, side))
    image = ax.imshow(cm, cmap="Blues")
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center", color="white" if cm[row, col] > threshold else "black")
    labels = [name.replace("_", " ").title() for name in class_names]
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel="Predicted", ylabel="Actual", title="Held-out test confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def class_weights(dataset, class_count):
    counts = Counter(int(row["label_id"]) for row in dataset.rows)
    total = sum(counts.values())
    values = [total / (class_count * max(1, counts[index])) for index in range(class_count)]
    return torch.tensor(values, dtype=torch.float32), counts


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = select_device(args.device)
    class_names = read_manifest_class_names(args.manifest)
    train_transform, eval_transform = build_transforms(args.image_size)
    datasets = {
        "train": ManifestDataset(args.manifest, "train", train_transform),
        "val": ManifestDataset(args.manifest, "val", eval_transform),
        "test": ManifestDataset(args.manifest, "test", eval_transform),
    }
    if any(len(dataset) == 0 for dataset in datasets.values()):
        raise SystemExit(f"Every split must contain images: { {key: len(value) for key, value in datasets.items()} }")
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"), num_workers=args.workers)
        for split, dataset in datasets.items()
    }

    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    model.to(device)
    weights, training_counts = class_weights(datasets["train"], len(class_names))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.05)
    history = []
    best_val_loss = math.inf
    best_epoch = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    start_time = time.time()

    phases = [("head", args.epochs_head, 1e-3), ("fine_tune", args.epochs_finetune, 2e-4)]
    epoch_number = 0
    for phase, epochs, learning_rate in phases:
        if phase == "fine_tune":
            for block in model.features[-3:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
        for _ in range(epochs):
            epoch_number += 1
            train_loss, train_accuracy = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)
            scheduler.step()
            record = {
                "epoch": epoch_number, "phase": phase,
                "train_loss": train_loss, "train_accuracy": train_accuracy,
                "val_loss": val_loss, "val_accuracy": val_accuracy,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch_number
                torch.save({
                    "state_dict": model.state_dict(), "class_names": class_names,
                    "image_size": args.image_size, "seed": args.seed, "task": args.task,
                }, checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    val_logits, val_labels, _ = collect_logits(model, loaders["val"], device)
    temperature = fit_temperature(val_logits, val_labels)
    test_logits, test_labels, test_paths = collect_logits(model, loaders["test"], device)
    probabilities = torch.softmax(test_logits / temperature, dim=1).numpy()
    labels = test_labels.numpy()
    predictions = probabilities.argmax(axis=1)
    class_ids = list(range(len(class_names)))
    cm = confusion_matrix(labels, predictions, labels=class_ids)
    ece, _ = expected_calibration_error(probabilities, labels)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(labels, predictions, labels=class_ids, average="macro", zero_division=0)
    report = classification_report(labels, predictions, labels=class_ids, target_names=class_names, output_dict=True, zero_division=0)
    metrics = {
        "task": args.task,
        "model": "MobileNetV3-Small transfer learning",
        "classes": class_names,
        "split_counts": {split: len(dataset) for split, dataset in datasets.items()},
        "training_class_counts": {class_names[index]: training_counts[index] for index in class_ids},
        "best_epoch": best_epoch,
        "training_seconds": time.time() - start_time,
        "device": str(device),
        "temperature": temperature,
        "test": {
            "accuracy": accuracy_score(labels, predictions),
            "balanced_accuracy": balanced_accuracy_score(labels, predictions),
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "log_loss": log_loss(labels, probabilities, labels=class_ids),
            "expected_calibration_error": ece,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        },
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    checkpoint["temperature"] = temperature
    torch.save(checkpoint, checkpoint_path)
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for path, actual, predicted, probs in zip(test_paths, labels, predictions, probabilities):
            handle.write(json.dumps({
                "image_path": path,
                "actual": class_names[int(actual)],
                "predicted": class_names[int(predicted)],
                "confidence": float(probs[predicted]),
                "probabilities": {name: float(prob) for name, prob in zip(class_names, probs)},
                "correct": bool(actual == predicted),
            }) + "\n")
    plot_training(history, args.output_dir)
    plot_confusion(cm, class_names, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
