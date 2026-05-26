from pathlib import Path

import torch
from torch import nn


SCAN_LABELS: dict[str, tuple[str, str]] = {
    "brain": ("no_tumor", "tumor"),
    "breast": ("benign_or_normal", "malignant_or_suspicious"),
}


class TumorClassifier(nn.Module):
    """Small CNN used for local inference and Grad-CAM.

    Production deployments should provide trained weights through
    ONCOSCAN_MODEL_PATH. The architecture is intentionally compact so the API
    can run in constrained containers and still expose a real Grad-CAM target.
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features)
        return self.classifier(pooled)

    @property
    def gradcam_target_layer(self) -> nn.Module:
        return self.features[8]


def load_model(model_path: Path | None, device: str) -> tuple[TumorClassifier, bool]:
    model = TumorClassifier()
    loaded = False

    if model_path and model_path.exists():
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        loaded = True

    model.to(device)
    model.eval()
    return model, loaded
