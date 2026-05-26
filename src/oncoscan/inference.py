import base64
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
from PIL import Image

from oncoscan.gradcam import GradCAM
from oncoscan.model import SCAN_LABELS, TumorClassifier

IMAGE_SIZE = 224
DISCLAIMER = (
    "OncoScan AI is a decision-support prototype and is not a medical device. "
    "Predictions must be reviewed by qualified clinicians."
)


@dataclass(frozen=True)
class InferenceResult:
    scan_type: str
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    heatmap_png_base64: str
    overlay_png_base64: str


def validate_scan_type(scan_type: str) -> str:
    normalized = scan_type.strip().lower()
    if normalized not in SCAN_LABELS:
        allowed = ", ".join(sorted(SCAN_LABELS))
        raise ValueError(f"scan_type must be one of: {allowed}")
    return normalized


def load_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Uploaded file must be a valid image.") from exc


def preprocess(image: Image.Image, device: str) -> torch.Tensor:
    resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device)


def infer(
    model: TumorClassifier,
    image_bytes: bytes,
    scan_type: str,
    device: str,
) -> InferenceResult:
    scan_type = validate_scan_type(scan_type)
    image = load_image(image_bytes)
    tensor = preprocess(image, device)

    labels = SCAN_LABELS[scan_type]
    with torch.enable_grad():
        with GradCAM(model, model.gradcam_target_layer) as gradcam:
            logits = model(tensor)
            probabilities_tensor = torch.softmax(logits, dim=1)[0]
            class_index = int(torch.argmax(probabilities_tensor).item())
            cam = gradcam.generate(tensor, class_index)

    probabilities = {
        label: round(float(probabilities_tensor[index].detach().cpu().item()), 6)
        for index, label in enumerate(labels)
    }
    confidence = probabilities[labels[class_index]]
    heatmap = render_heatmap(cam)
    overlay = render_overlay(image, heatmap)

    return InferenceResult(
        scan_type=scan_type,
        prediction=labels[class_index],
        confidence=confidence,
        probabilities=probabilities,
        heatmap_png_base64=encode_png(heatmap),
        overlay_png_base64=encode_png(overlay),
    )


def render_heatmap(cam: np.ndarray) -> Image.Image:
    heat = Image.fromarray(np.uint8(cam * 255), mode="L").resize((IMAGE_SIZE, IMAGE_SIZE))
    heat_array = np.asarray(heat, dtype=np.float32) / 255.0

    red = np.clip(2.0 * heat_array, 0, 1)
    green = np.clip(2.0 * (1.0 - np.abs(heat_array - 0.5)), 0, 1)
    blue = np.clip(2.0 * (1.0 - heat_array), 0, 1) * 0.25
    rgb = np.stack([red, green, blue], axis=-1)
    return Image.fromarray(np.uint8(rgb * 255), mode="RGB")


def render_overlay(image: Image.Image, heatmap: Image.Image) -> Image.Image:
    base = image.resize((IMAGE_SIZE, IMAGE_SIZE)).convert("RGB")
    return Image.blend(base, heatmap, alpha=0.42)


def encode_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
