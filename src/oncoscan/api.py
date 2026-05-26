from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from oncoscan.config import get_settings
from oncoscan.inference import DISCLAIMER, infer
from oncoscan.model import load_model
from oncoscan.schemas import HealthResponse, PredictionResponse


def create_app() -> FastAPI:
    settings = get_settings()
    state = {}
    static_dir = Path(__file__).resolve().parent / "static"

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["model"], state["model_loaded"] = load_model(settings.model_path, settings.device)
        yield

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Tumor detection API with Grad-CAM explainability for brain MRI and breast scans.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        with open(static_dir / "index.html", encoding="utf-8") as page:
            return page.read()

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            model_loaded=bool(state.get("model_loaded", False)),
            device=settings.device,
            environment=settings.environment,
        )

    @app.post("/api/v1/predict", response_model=PredictionResponse)
    async def predict(
        scan_type: str = Form(..., description="brain or breast"),
        image: UploadFile = File(...),
    ) -> PredictionResponse:
        if image.content_type and not image.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Only image uploads are supported.")

        image_bytes = await image.read()
        max_bytes = settings.max_upload_mb * 1024 * 1024
        if len(image_bytes) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Image exceeds {settings.max_upload_mb} MB limit.")

        try:
            result = infer(state["model"], image_bytes, scan_type, settings.device)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        return PredictionResponse(
            scan_type=result.scan_type,
            prediction=result.prediction,
            confidence=result.confidence,
            probabilities=result.probabilities,
            heatmap_png_base64=result.heatmap_png_base64,
            overlay_png_base64=result.overlay_png_base64,
            model_loaded=bool(state.get("model_loaded", False)),
            disclaimer=DISCLAIMER,
        )

    return app
