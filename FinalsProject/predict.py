"""
predict.py  —  Classify a single image (CLI)
Usage: python predict.py path/to/image.jpg
"""

import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "models/cat_dog_model.pth"
IMG_SIZE   = 160
LABELS     = {0: "Cat", 1: "Dog"}

infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def predict(img_path: str) -> None:
    if not os.path.exists(MODEL_PATH):
        print(f"\n  Model not found at '{MODEL_PATH}'.")
        print("  Run 'python train.py' first.\n")
        return
    if not os.path.exists(img_path):
        print(f"\n  Image not found: {img_path}\n")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 1),
        nn.Sigmoid(),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    img    = Image.open(img_path).convert("RGB")
    tensor = infer_tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = float(model(tensor)[0][0])

    label = LABELS[int(prob >= 0.5)]
    conf  = prob if prob >= 0.5 else 1 - prob
    icon  = "🐱" if label == "Cat" else "🐶"

    print(f"\n  {icon}  Prediction : {label}")
    print(f"      Confidence : {conf * 100:.1f}%\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("\n  Usage: python predict.py path/to/image.jpg\n")
    else:
        predict(sys.argv[1])
