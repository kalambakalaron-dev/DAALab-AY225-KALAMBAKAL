# Cat vs Dog Classifier (PyTorch)

Dark-themed web dashboard · Flask + PyTorch ResNet18 · Python 3.13+

---

## Quick Start in VS Code

### Step 1 — Open project in VS Code
1. Extract the `cat-dog-classifier` folder anywhere (e.g. `C:\Users\Aaron\Projects\`)
2. Open VS Code → **File → Open Folder** → select `cat-dog-classifier`
3. Open terminal: **Ctrl + `** (backtick)
   > You are now automatically in the correct folder — no `cd` needed!

### Step 2 — Install PyTorch
PyTorch needs to be installed separately first:

**CPU only (works on any PC):**
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**With NVIDIA GPU (faster training):**
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest:
```
pip install -r requirements.txt
```

### Step 3 — Download the Kaggle dataset

**Manual (easiest):**
1. Go to: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification
2. Click Download (free Kaggle account needed)
3. Extract and arrange like this:
```
dataset/
└── train/
    ├── cats/   ← cat images here
    └── dogs/   ← dog images here
```

**Via Kaggle API:**
1. https://www.kaggle.com/settings → API → Create New Token → saves kaggle.json
2. Move `kaggle.json` to `C:\Users\Aaron\.kaggle\kaggle.json`
3. Then:
```
kaggle datasets download -d samuelcortinhas/cats-and-dogs-image-classification
```
Extract the ZIP into `dataset/train/`

### Step 4 — Train
```
python train.py
```
Saves model to `models/cat_dog_model.pth`

### Step 5 — Run the web app
```
python app.py
```
Open: **http://127.0.0.1:5500**

### Step 6 — CLI prediction (optional)
```
python predict.py C:\Users\Aaron\Pictures\mycat.jpg
```

---

## Push to GitHub
```
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/cat-dog-classifier.git
git push -u origin main
```

---

## Common Errors

| Error | Fix |
|-------|-----|
| `No such file: requirements.txt` | Open folder in VS Code first, use VS Code terminal |
| `Model not found` | Run `python train.py` first |
| `No module named torch` | Run the pip install torch command in Step 2 |
| `0 images found` | Check images are inside `dataset/train/cats/` and `dataset/train/dogs/` |
| Port 5500 in use | Change `port=5500` in app.py to `5501` |

---

## Architecture

- **Model:** ResNet18 pretrained on ImageNet (frozen), custom head
- **Head:** Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1) → Sigmoid
- **Optimizer:** Adam, lr=0.001
- **Loss:** Binary Cross Entropy
- **Classes:** cats=0, dogs=1
