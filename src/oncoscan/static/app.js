const form = document.querySelector("#predict-form");
const statusBox = document.querySelector("#status");
const results = document.querySelector("#results");
const prediction = document.querySelector("#prediction");
const confidence = document.querySelector("#confidence");
const heatmap = document.querySelector("#heatmap");
const overlay = document.querySelector("#overlay");
const probabilities = document.querySelector("#probabilities");
const disclaimer = document.querySelector("#disclaimer");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  statusBox.textContent = "Analyzing image and generating Grad-CAM evidence...";
  results.classList.add("hidden");

  try {
    const response = await fetch("/api/v1/predict", {
      method: "POST",
      body: new FormData(form),
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Prediction failed.");
    }

    prediction.textContent = payload.prediction.replaceAll("_", " ");
    confidence.textContent = `${Math.round(payload.confidence * 100)}%`;
    heatmap.src = `data:image/png;base64,${payload.heatmap_png_base64}`;
    overlay.src = `data:image/png;base64,${payload.overlay_png_base64}`;
    probabilities.textContent = JSON.stringify(payload.probabilities, null, 2);
    disclaimer.textContent = payload.model_loaded
      ? payload.disclaimer
      : `${payload.disclaimer} No trained weight file is loaded, so this is running in demo mode.`;
    results.classList.remove("hidden");
    statusBox.textContent = "Analysis complete.";
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
