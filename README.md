# Cat vs Dog Classifier — PyTorch (Python 3.13 compatible)

## Setup (do this once)

### Step 1 — Install libraries
Open terminal in VS Code (`Ctrl + backtick`) and run:
```
pip install torch torchvision matplotlib seaborn scikit-learn Pillow numpy
```
Wait for it to finish (2–5 minutes).

### Step 2 — Download the dataset
1. Go to: https://www.kaggle.com/datasets/samuelcortinhas/cats-and-dogs-image-classification
2. Click Download (you need a Kaggle account)
3. Extract the ZIP file
4. Create a folder called `dataset` in the same folder as `train.py`
5. Inside `dataset`, put the `train` folder from the ZIP

Your folder structure should look like this:
```
cat-dog-classifier/
├── train.py
├── predict.py
├── requirements.txt
└── dataset/
    └── train/
        ├── cats/
        │   ├── cat1.jpg
        │   └── ...
        └── dogs/
            ├── dog1.jpg
            └── ...
```

### Step 3 — Train the model
In VS Code terminal:
```
python train.py
```
Training takes about 10–20 minutes. You will see accuracy printed each epoch.
When done it saves `cat_dog_model.pth` and shows accuracy/confusion matrix charts.

### Step 4 — Test with your own image
```
python predict.py path/to/any/cat/or/dog/image.jpg
```

## Expected Results
- ~92–97% accuracy after 5 epochs
- Accuracy/loss charts saved to `plots/training_curves.png`
- Confusion matrix saved to `plots/confusion_matrix.png`

## Why PyTorch instead of TensorFlow?
TensorFlow does not support Python 3.13. PyTorch does.

## Team
| Role | Name |
|------|------|
| Repo owner | @Josh Aron Kalambakal |
| Collaborator | @Jazen Kierr Malano|
