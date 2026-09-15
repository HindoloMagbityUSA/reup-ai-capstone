from __future__ import annotations

import hashlib
import io
import threading
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from reup_ai.modeling import inference_transform, load_classifier


MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000


@dataclass
class LoadedClassifier:
    task: str
    checkpoint_path: Path
    threshold: float

    def __post_init__(self) -> None:
        self.model, self.checkpoint = load_classifier(self.checkpoint_path)
        self.class_names = list(self.checkpoint["class_names"])
        self.transform = inference_transform(int(self.checkpoint["image_size"]))
        self.temperature = float(self.checkpoint.get("temperature", 1.0))
        self._lock = threading.Lock()

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image).unsqueeze(0)
        with self._lock, torch.inference_mode():
            probabilities = torch.softmax(
                self.model(tensor) / self.temperature, dim=1
            ).squeeze(0)

        confidence, prediction = probabilities.max(0)
        raw_prediction = self.class_names[prediction.item()]
        probability_map = {
            name: float(probability.item())
            for name, probability in zip(self.class_names, probabilities)
        }
        ranked = sorted(
            probability_map.items(), key=lambda item: item[1], reverse=True
        )
        accepted = float(confidence.item()) >= self.threshold
        return {
            "status": "suggested" if accepted else "low_confidence",
            "suggestion": raw_prediction if accepted else "unable_to_determine",
            "raw_prediction": raw_prediction,
            "confidence": float(confidence.item()),
            "confidence_threshold": self.threshold,
            "top_alternatives": [
                {"label": label, "probability": probability}
                for label, probability in ranked[:3]
            ],
            "probabilities": probability_map,
            "verification_required": True,
        }


class ImageInferenceService:
    def __init__(self, model_paths: dict[str, Path], thresholds: dict[str, float]):
        missing = [str(path) for path in model_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Required model checkpoints are missing: {missing}")
        self.classifiers = {
            task: LoadedClassifier(task, path, thresholds[task])
            for task, path in model_paths.items()
        }

    @staticmethod
    def decode_image(content: bytes) -> Image.Image:
        if not content:
            raise ValueError("The uploaded image is empty.")
        if len(content) > MAX_IMAGE_BYTES:
            raise ValueError("The uploaded image exceeds the 12 MB demo limit.")
        try:
            image = Image.open(io.BytesIO(content))
            image.verify()
            image = Image.open(io.BytesIO(content))
            image = ImageOps.exif_transpose(image).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file is not a valid JPEG or PNG image.") from exc
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise ValueError("The uploaded image dimensions are too large for the demo.")
        return image

    def analyze(self, content: bytes) -> dict:
        image = self.decode_image(content)
        return {
            "image_sha256": hashlib.sha256(content).hexdigest(),
            "image_width": image.width,
            "image_height": image.height,
            "verification_required": True,
            "structural_or_safety_assessment": False,
            "fields": {
                task: classifier.predict(image)
                for task, classifier in self.classifiers.items()
            },
        }

    def model_status(self) -> dict:
        return {
            task: {
                "loaded": True,
                "checkpoint": classifier.checkpoint_path.name,
                "classes": classifier.class_names,
                "confidence_threshold": classifier.threshold,
            }
            for task, classifier in self.classifiers.items()
        }
