## END to END Tumor Detection with XAI and GCP

# OncoScan AI is a full-stack medical imaging application for detecting tumors in MRI (Brain) and Breast Scan images. It leverages Explainable AI (XAI) with GradCAM to provide heatmaps highlighting regions influencing AI's predictions, crucial for medical professionals, and features a production-grade MLOps pipeline.

# System Design
![System Architecture][./assets/system_design_oncoscan.png]

# what's need to build
A tumor detection system for MRI and Breast Scan images.
A REST API endpoint for receiving image data and returning predictions.
A GradCAM heatmap visualization tool integrated into the frontend.
A Dockerized application for consistent execution across environments.
A Kubernetes deployment for scalable and highly available service.
