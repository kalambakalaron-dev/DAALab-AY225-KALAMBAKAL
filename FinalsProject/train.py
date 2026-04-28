"""
train.py  —  Train Cat vs Dog classifier with PyTorch
Usage: python train.py
"""

import os
import sys
import copy
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR   = "dataset/train"
MODEL_PATH = "models/cat_dog_model.pth"
IMG_SIZE   = 160
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-3

os.makedirs("models", exist_ok=True)
os.makedirs("plots",  exist_ok=True)

# ── Validate dataset ───────────────────────────────────────────────────────────
for cls in ["cats", "dogs"]:
    p = os.path.join(DATA_DIR, cls)
    if not os.path.isdir(p):
        print(f"\n  Missing folder: {p}")
        print("  Please set up your dataset first.")
        print("  See README.md -> Step 3 for Kaggle download instructions.\n")
        sys.exit(1)

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ── Transforms ─────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Dataset split ──────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
print(f"Class mapping: {full_dataset.class_to_idx}")   # {'cats': 0, 'dogs': 1}

val_size   = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

# Apply val transforms to val split
val_ds.dataset = copy.deepcopy(full_dataset)
val_ds.dataset.transform = val_tf

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {train_size} images  |  Val: {val_size} images\n")

# ── Model (ResNet18 transfer learning) ────────────────────────────────────────
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False
# Replace final layer for binary classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)
model = model.to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

# ── Training loop ──────────────────────────────────────────────────────────────
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
best_val_acc  = 0.0
best_weights  = None
patience_ctr  = 0
PATIENCE      = 3

print("Training started...\n")

for epoch in range(EPOCHS):
    t0 = time.time()

    # — Train —
    model.train()
    t_loss, t_correct = 0.0, 0
    for imgs, labels in train_loader:
        imgs   = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * imgs.size(0)
        t_correct += ((out >= 0.5).float() == labels).sum().item()

    # — Validate —
    model.eval()
    v_loss, v_correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            out    = model(imgs)
            loss   = criterion(out, labels)
            v_loss    += loss.item() * imgs.size(0)
            v_correct += ((out >= 0.5).float() == labels).sum().item()

    t_acc = t_correct / train_size
    v_acc = v_correct / val_size
    t_l   = t_loss    / train_size
    v_l   = v_loss    / val_size

    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)
    history["train_loss"].append(t_l)
    history["val_loss"].append(v_l)

    elapsed = time.time() - t0
    print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
          f"train_loss={t_l:.4f}  train_acc={t_acc:.4f}  "
          f"val_loss={v_l:.4f}  val_acc={v_acc:.4f}  "
          f"({elapsed:.1f}s)")

    scheduler.step()

    # Early stopping
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        best_weights = copy.deepcopy(model.state_dict())
        torch.save(best_weights, MODEL_PATH)
        print(f"           Best model saved (val_acc={best_val_acc:.4f})")
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"\nEarly stopping after {epoch+1} epochs.")
            break

# ── Save training curves ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor("#0f0f1a")
for ax in axes:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

ep = range(1, len(history["train_acc"]) + 1)
axes[0].plot(ep, [a * 100 for a in history["train_acc"]], color="#7f77dd", label="Train")
axes[0].plot(ep, [a * 100 for a in history["val_acc"]],   color="#1d9e75", label="Val")
axes[0].set_title("Accuracy (%)", color="white")
axes[0].legend(facecolor="#1a1a2e", labelcolor="white")

axes[1].plot(ep, history["train_loss"], color="#7f77dd", label="Train")
axes[1].plot(ep, history["val_loss"],   color="#1d9e75", label="Val")
axes[1].set_title("Loss", color="white")
axes[1].legend(facecolor="#1a1a2e", labelcolor="white")

plt.tight_layout()
plt.savefig("plots/training_curves.png", facecolor=fig.get_facecolor())

print(f"\nBest val accuracy : {best_val_acc * 100:.2f}%")
print(f"Model saved       : {MODEL_PATH}")
print(f"Training curves   : plots/training_curves.png\n")
