"""Build the file inventory, audit patient leakage, and write patient-grouped splits.

Run from the repo root:  python -m src.make_splits
"""
import re
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from .utils import ROOT, save_json

SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "results"


def find_data_root():
    base = ROOT / "data" / "raw" / "chest_xray"
    nested = base / "chest_xray"
    return nested if (nested / "train").exists() else base


def patient_id(path):
    name = Path(path).name
    m = re.match(r"(person\d+)", name)
    if m:
        return m.group(1)
    m = re.search(r"(IM-\d+)", name)
    if m:
        return m.group(1)
    return name  # fallback: image is its own group


def build_inventory(root=None):
    root = Path(root) if root else find_data_root()
    rows = []
    for split in ["train", "val", "test"]:
        for label in ["NORMAL", "PNEUMONIA"]:
            for p in sorted((root / split / label).glob("*.jp*g")):
                if p.name.startswith("."):      # skip macOS ._ resource files
                    continue
                with Image.open(p) as im:
                    w, h = im.size
                    mode = im.mode
                rows.append(dict(path=str(p), split=split, label=label, w=w, h=h, mode=mode))
    df = pd.DataFrame(rows)
    df["y"] = (df.label == "PNEUMONIA").astype(int)
    df["group"] = df.path.map(patient_id)
    return df


def main(seed=42):
    df = build_inventory()
    print(len(df), "images found")
    assert len(df) > 5000, "Expected ~5,856 images. Check data/raw/chest_xray path."

    # ---- leakage audit on the OFFICIAL splits
    sets = {s: set(df[df.split == s].group) for s in ["train", "val", "test"]}
    audit = {
        "n_images": len(df),
        "n_patient_groups": int(df.group.nunique()),
        "unparsed_filenames": int((df.group == df.path.map(lambda p: Path(p).name)).sum()),
        "official_overlap_train_val": len(sets["train"] & sets["val"]),
        "official_overlap_train_test": len(sets["train"] & sets["test"]),
        "official_overlap_val_test": len(sets["val"] & sets["test"]),
    }
    print("Audit:", audit)

    # ---- patient-grouped re-split of train+val (test stays held out)
    pool = df[df.split != "test"].copy()
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(sgkf.split(pool, pool.y, pool.group))
    train_df, val_df = pool.iloc[tr_idx], pool.iloc[va_idx]
    test_df = df[df.split == "test"]

    assert set(train_df.group).isdisjoint(val_df.group), "Patient leakage train/val!"
    assert set(train_df.path).isdisjoint(val_df.path), "Image overlap train/val!"

    SPLITS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    train_df.to_csv(SPLITS / "train.csv", index=False)
    val_df.to_csv(SPLITS / "val.csv", index=False)
    test_df.to_csv(SPLITS / "test.csv", index=False)

    summary = pd.DataFrame({
        "n_images": [len(train_df), len(val_df), len(test_df)],
        "pneumonia_pct": [train_df.y.mean() * 100, val_df.y.mean() * 100, test_df.y.mean() * 100],
        "patient_groups": [train_df.group.nunique(), val_df.group.nunique(), test_df.group.nunique()],
    }, index=["train", "val", "test"]).round(1)
    summary.to_csv(RESULTS / "split_summary.csv")
    audit["pos_weight_train"] = float((train_df.y == 0).sum() / (train_df.y == 1).sum())
    save_json(audit, RESULTS / "leakage_audit.json")
    print(summary)
    print("pos_weight (n_neg/n_pos):", round(audit["pos_weight_train"], 3))
    print("Splits written to", SPLITS)


if __name__ == "__main__":
    main()
