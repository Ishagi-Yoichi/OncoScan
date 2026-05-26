# OncoScan AI

OncoScan AI is a full-stack prototype for tumor detection in brain MRI and breast scan images. It exposes a REST API, a browser upload interface, and Grad-CAM visual explanations for model predictions.

This repository does not include a trained clinical model or medical dataset. If `ONCOSCAN_MODEL_PATH` is not set, the service runs with randomly initialized demo weights and marks responses as demo mode. Do not use demo outputs for diagnosis.

## Features

- FastAPI service with `POST /api/v1/predict` for image uploads.
- Brain and breast scan label spaces.
- Grad-CAM heatmap and overlay returned as base64 PNGs.
- Static frontend at `/` for local uploads and visualization.
- Dockerfile and Kubernetes manifests for deployment.
- CI workflow for linting and tests.

## Local Setup

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python main.py
```

Open `http://localhost:8000`.

## Configuration

Environment variables use the `ONCOSCAN_` prefix:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ONCOSCAN_MODEL_PATH` | unset | Path to a PyTorch `state_dict` for `TumorClassifier`. |
| `ONCOSCAN_DEVICE` | `cpu` | Runtime device, for example `cpu` or `cuda`. |
| `ONCOSCAN_MAX_UPLOAD_MB` | `10` | Upload size limit. |
| `ONCOSCAN_ENVIRONMENT` | `development` | Environment name surfaced by health checks. |
| `ONCOSCAN_CORS_ORIGINS` | `["*"]` | CORS origins parsed by Pydantic settings. |

## API

Health check:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Prediction:

```powershell
curl -X POST http://localhost:8000/api/v1/predict ^
  -F "scan_type=brain" ^
  -F "image=@sample.png"
```

The response includes `prediction`, `confidence`, class `probabilities`, `heatmap_png_base64`, `overlay_png_base64`, `model_loaded`, and a clinical disclaimer.

## Docker

```powershell
docker build -t oncoscan:latest .
docker run --rm -p 8000:8000 -e ONCOSCAN_ENVIRONMENT=container oncoscan:latest
```

Mount trained weights when available:

```powershell
docker run --rm -p 8000:8000 ^
  -v ${PWD}\models:/models ^
  -e ONCOSCAN_MODEL_PATH=/models/tumor_classifier.pt ^
  oncoscan:latest
```

## Kubernetes

```powershell
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml
```

Update the image name in `k8s/deployment.yaml` for your registry. Store model weights in a persistent volume, object storage init container, or model registry integration before production rollout.

## GCP Cloud Build

`cloudbuild.yaml` builds the container, pushes it to Artifact Registry, and applies the Kubernetes manifests to GKE. Set the substitutions for your project, region, repository, cluster, and zone.

```powershell
gcloud builds submit --config cloudbuild.yaml ^
  --substitutions _REGION=us-central1,_REPOSITORY=oncoscan,_CLUSTER=oncoscan-gke,_ZONE=us-central1-a
```

## Model Training Path

The included `TumorClassifier` is a compact CNN intended to make inference and Grad-CAM wiring executable. For real use, train and validate on governed datasets, export the matching `state_dict`, and deploy with `ONCOSCAN_MODEL_PATH`.

Recommended production gates:

- Dataset versioning and bias review.
- Train/validation/test split with patient-level separation.
- Sensitivity, specificity, ROC-AUC, calibration, and subgroup metrics.
- Human review workflow and audit logging.
- Model registry approval before Kubernetes rollout.
