import os
import joblib
import numpy as np
import requests
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# ✅ Model URL and Path
MODEL_URL = "https://huggingface.co/UmeshSamartapu/NewProject/resolve/main/rf_model.pkl"
MODEL_PATH = "rf_model.pkl"

# ✅ Download model if not already present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model...")
        r = requests.get(MODEL_URL)
        if r.status_code == 200:
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
            print("✅ Model downloaded successfully.")
        else:
            print("❌ Failed to download model:", r.status_code)
            raise Exception("Model download failed.")

download_model()

# ✅ Load model
model = joblib.load(MODEL_PATH)

# ✅ Class label mapping
class_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# ✅ FastAPI setup
app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    f1: float = Form(...),
    f2: float = Form(...),
    f3: float = Form(...),
    f4: float = Form(...)
):
    features = np.array([[f1, f2, f3, f4]])
    prediction = model.predict(features)[0]
    flower_name = class_names.get(prediction, "Unknown")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": f"🌼 Predicted Class: {flower_name} ({prediction})"
    })
