from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    device: str
    environment: str


class PredictionResponse(BaseModel):
    scan_type: str
    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]
    heatmap_png_base64: str
    overlay_png_base64: str
    model_loaded: bool
    disclaimer: str
