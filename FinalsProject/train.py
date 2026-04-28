"""
Cat vs Dog Classifier — PyTorch + EfficientNet
Works on Python 3.13 | Run: python train.py
"""

import os, time, copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── 1. Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "dataset/train"   # folder with cats/ and dogs/ subfolders
MODEL_PATH = "cat_dog_model.pth"
IMG_SIZE   = 224
BATCH_SIZE = 32
EPOCHS     = 5                 # 5 is fast and accurate enough; raise to 10+ for better results
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print(f"Data folder : {DATA_DIR}")

# ── 2. Check dataset exists ────────────────────────────────────────────────────
if not os.path.exists(DATA_DIR):
    print(f"\nERROR: '{DATA_DIR}' folder not found.")
    print("Please make sure your dataset is set up like this:")
    print("  dataset/train/cats/  <-- put cat images here")
    print("  dataset/train/dogs/  <-- put dog images here")
    exit(1)

# ── 3. Data transforms ─────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── 4. Load dataset ────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATA_DIR)
class_names  = full_dataset.classes
print(f"Classes found: {class_names}")
print(f"Total images : {len(full_dataset)}")

val_size   = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])

train_set.dataset.transform = train_transforms
val_set.dataset.transform   = val_transforms

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {train_size} images | Val: {val_size} images")

# ── 5. Model (EfficientNet_B0 transfer learning) ───────────────────────────────
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# Freeze all pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace final classifier layer for binary classification
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.classifier[1].in_features, 2)
)

model = model.to(DEVICE)
print("Model ready (EfficientNet_B0 pretrained)")

# ── 6. Loss & optimizer ────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ── 7. Training loop ───────────────────────────────────────────────────────────
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
best_acc   = 0.0
best_model = copy.deepcopy(model.state_dict())

print("\nStarting training...")
print("-" * 50)

for epoch in range(EPOCHS):
    start = time.time()

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
            loader = train_loader
        else:
            model.eval()
            loader = val_loader

        running_loss, running_correct = 0.0, 0

        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                preds   = outputs.argmax(dim=1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss    += loss.item() * inputs.size(0)
            running_correct += (preds == labels).sum().item()

        size     = train_size if phase == "train" else val_size
        epoch_loss = running_loss    / size
        epoch_acc  = running_correct / size

        history[f"{phase}_loss"].append(epoch_loss)
        history[f"{phase}_acc"].append(epoch_acc)

        if phase == "val" and epoch_acc > best_acc:
            best_acc   = epoch_acc
            best_model = copy.deepcopy(model.state_dict())

    scheduler.step()
    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train acc: {history['train_acc'][-1]*100:.1f}% | "
          f"Val acc: {history['val_acc'][-1]*100:.1f}% | "
          f"Time: {elapsed:.0f}s")

print(f"\nBest validation accuracy: {best_acc*100:.1f}%")

# ── 8. Save best model ─────────────────────────────────────────────────────────
torch.save({
    "model_state": best_model,
    "class_names": class_names,
    "img_size":    IMG_SIZE
}, MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")

# ── 9. Accuracy & loss plots ───────────────────────────────────────────────────
os.makedirs("plots", exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot([a*100 for a in history["train_acc"]], label="Train")
axes[0].plot([a*100 for a in history["val_acc"]],   label="Validation")
axes[0].set_title("Accuracy per Epoch")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
axes[0].legend()

axes[1].plot(history["train_loss"], label="Train")
axes[1].plot(history["val_loss"],   label="Validation")
axes[1].set_title("Loss per Epoch")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("plots/training_curves.png")
plt.show()
print("Training curves saved → plots/training_curves.png")

# ── 10. Confusion matrix ───────────────────────────────────────────────────────
model.load_state_dict(best_model)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png")
plt.show()

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))
print("\nDone! All outputs saved in plots/")
