"""
Predict a single image — Cat or Dog
Usage: python predict.py path/to/image.jpg
"""

import sys, os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "cat_dog_model.pth"

def load_model(path):
    checkpoint  = torch.load(path, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    img_size    = checkpoint["img_size"]

    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names, img_size

def predict(img_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found. Run train.py first.")
        return

    model, class_names, img_size = load_model(MODEL_PATH)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img    = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = probs.argmax().item()

    print(f"Prediction : {class_names[pred].upper()}")
    print(f"Confidence : {probs[pred].item()*100:.1f}%")
    print(f"  cats: {probs[0].item()*100:.1f}%")
    print(f"  dogs: {probs[1].item()*100:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
    else:
        predict(sys.argv[1])
