"""
app.py  —  Cat vs Dog Classifier  (Flask + PyTorch)
Usage:  python app.py
Open:   http://127.0.0.1:5500
"""

import os
import io
import json
import base64
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/cat_dog_model.pth"
IMG_SIZE   = 160
LABELS     = {0: "Cat", 1: "Dog"}
STATS_FILE = "static/prediction_stats.json"

Path("static/uploads").mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

# ── Transform (same as val in train.py) ───────────────────────────────────────
infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Model loader ───────────────────────────────────────────────────────────────
_model  = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    global _model
    if _model is not None:
        return _model
    if not os.path.exists(MODEL_PATH):
        return None

    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    m.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
    m.to(_device)
    m.eval()
    _model = m
    return _model

# ── Stats helpers ──────────────────────────────────────────────────────────────
def load_stats():
    if os.path.exists(STATS_FILE):
        with open(STATS_FILE) as f:
            return json.load(f)
    return {"total": 0, "cats": 0, "dogs": 0, "history": []}

def save_stats(s):
    with open(STATS_FILE, "w") as f:
        json.dump(s, f)

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template(
        "index.html",
        model_ready=os.path.exists(MODEL_PATH),
        stats=load_stats(),
    )

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if Path(file.filename).suffix.lower() not in allowed:
        return jsonify({"error": "Unsupported file type"}), 400

    model = get_model()
    if model is None:
        return jsonify({"error": "Model not found. Run python train.py first."}), 503

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Thumbnail for preview
        thumb = img.copy()
        thumb.thumbnail((300, 300))
        buf = io.BytesIO()
        thumb.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Inference
        tensor = infer_tf(img).unsqueeze(0).to(_device)
        with torch.no_grad():
            prob = float(model(tensor)[0][0])

        label = LABELS[int(prob >= 0.5)]
        conf  = prob if prob >= 0.5 else 1 - prob

        # Update stats
        stats = load_stats()
        stats["total"] += 1
        stats["cats" if label == "Cat" else "dogs"] += 1
        stats["history"].append({"label": label, "confidence": round(conf * 100, 1)})
        stats["history"] = stats["history"][-20:]
        save_stats(stats)

        return jsonify({
            "label":      label,
            "confidence": round(conf * 100, 1),
            "prob_dog":   round(prob * 100, 1),
            "prob_cat":   round((1 - prob) * 100, 1),
            "image_b64":  img_b64,
            "stats":      stats,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stats")
def stats():
    return jsonify(load_stats())

@app.route("/model-status")
def model_status():
    return jsonify({"ready": os.path.exists(MODEL_PATH)})

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  Cat vs Dog Classifier (PyTorch)")
    print("  --------------------------------")
    print(f"  Device : {_device}")
    if os.path.exists(MODEL_PATH):
        print(f"  Model  : {MODEL_PATH}  [READY]")
    else:
        print(f"  Model  : NOT FOUND  ->  run 'python train.py' first")
    print("\n  Open:  http://127.0.0.1:5500\n")
    app.run(debug=True, host="127.0.0.1", port=5500)
