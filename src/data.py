from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Mild, anatomically plausible augmentation. No flips (heart is on the patient's left).
train_tf = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

eval_tf = T.Compose([
    T.Grayscale(num_output_channels=3),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


class CXRDataset(Dataset):
    def __init__(self, csv_path, tf):
        df = pd.read_csv(csv_path)
        self.paths = df["path"].tolist()
        self.labels = df["y"].astype("float32").tolist()
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("L")
        return self.tf(img), torch.tensor(self.labels[i], dtype=torch.float32)


def get_loaders(bs=32, workers=0):
    tr = CXRDataset(SPLITS / "train.csv", train_tf)
    va = CXRDataset(SPLITS / "val.csv", eval_tf)
    te = CXRDataset(SPLITS / "test.csv", eval_tf)
    return (
        DataLoader(tr, bs, shuffle=True, num_workers=workers),
        DataLoader(va, bs, shuffle=False, num_workers=workers),
        DataLoader(te, bs, shuffle=False, num_workers=workers),
    )


def get_pos_weight():
    """n_negative / n_positive from the TRAIN split, float32 (MPS has no float64)."""
    y = pd.read_csv(SPLITS / "train.csv")["y"]
    return torch.tensor([(y == 0).sum() / (y == 1).sum()], dtype=torch.float32)
