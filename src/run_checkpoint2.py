"""Runs every Checkpoint 2 experiment on VALIDATION data only (test set is never loaded).

    python -m src.run_checkpoint2              # full run (~20-40 min on Apple Silicon)
    python -m src.run_checkpoint2 --fast       # quick pipeline check (1 epoch each)
    python -m src.run_checkpoint2 --force      # redo stages that already have results

Stages are resumable: finished stages (results/*.json exist) are skipped.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .data import SPLITS, get_loaders, get_pos_weight
from .evaluate import compute_metrics, predict
from .models import SimpleCNN, build_resnet18, unfreeze_all
from .train import fit
from .utils import ROOT, get_device, save_json, seed_everything

RESULTS = ROOT / "results"
CKPT = RESULTS / "checkpoints"


def plot_curves(hist, title, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist.epoch, hist.train_loss, marker="o", label="train loss")
    ax[0].plot(hist.epoch, hist.val_loss, marker="o", label="val loss")
    ax[0].set_xlabel("epoch"); ax[0].set_title(f"{title}: loss"); ax[0].legend()
    for col, lab in [("val_auc", "AUC"), ("val_recall", "recall"), ("val_specificity", "specificity")]:
        ax[1].plot(hist.epoch, hist[col], marker="o", label=lab)
    ax[1].set_xlabel("epoch"); ax[1].set_ylim(0, 1.02)
    ax[1].set_title(f"{title}: validation metrics"); ax[1].legend()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def stage_dummy():
    val_df = pd.read_csv(SPLITS / "val.csv")
    m = compute_metrics(val_df.y.values, np.ones(len(val_df)))  # always predict pneumonia
    save_json(m, RESULTS / "dummy_val.json")
    print("Majority-class dummy:", {k: round(v, 3) for k, v in m.items() if isinstance(v, float)})


def stage_simplecnn(a, device, tl, vl, criterion):
    if (RESULTS / "simplecnn_val.json").exists() and not a.force:
        print("SimpleCNN already done, skipping"); return
    print("\n=== SimpleCNN baseline ===", flush=True)
    seed_everything(42)
    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist, best = fit(model, tl, vl, criterion, opt, device, a.simple_epochs, CKPT / "simplecnn.pt")
    hist.to_csv(RESULTS / "simplecnn_history.csv", index=False)
    save_json(best, RESULTS / "simplecnn_val.json")
    plot_curves(hist, "SimpleCNN", RESULTS / "simplecnn_curves.png")
    print("SimpleCNN best:", {k: round(v, 3) for k, v in best.items() if isinstance(v, float)})


def stage_resnet(a, device, tl, vl, criterion):
    if (RESULTS / "resnet18_val.json").exists() and not a.force:
        print("ResNet18 already done, skipping"); return
    seed_everything(42)
    model = build_resnet18(freeze_backbone=True).to(device)

    print("\n=== ResNet18 phase 1: frozen backbone, train head ===", flush=True)
    opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    h1, b1 = fit(model, tl, vl, criterion, opt, device, a.p1_epochs, CKPT / "resnet18_phase1.pt")

    print("\n=== ResNet18 phase 2: unfreeze all, backbone lr 1e-4, head lr 1e-3 ===", flush=True)
    unfreeze_all(model)
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc")]
    opt = torch.optim.Adam([
        {"params": backbone, "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    h2, b2 = fit(model, tl, vl, criterion, opt, device, a.p2_epochs, CKPT / "resnet18_phase2.pt", scheduler=sched)

    h1["phase"], h2["phase"] = 1, 2
    h2["epoch"] += len(h1)
    hist = pd.concat([h1, h2], ignore_index=True)
    hist.to_csv(RESULTS / "resnet18_history.csv", index=False)
    plot_curves(hist, "ResNet18", RESULTS / "resnet18_curves.png")

    # keep whichever phase had the best validation AUC (report honestly)
    if b2["auc"] >= b1["auc"]:
        best, src_ck, phase = b2, CKPT / "resnet18_phase2.pt", 2
    else:
        best, src_ck, phase = b1, CKPT / "resnet18_phase1.pt", 1
    best = {**best, "phase": phase}
    shutil.copy(src_ck, CKPT / "resnet18_best.pt")
    save_json(best, RESULTS / "resnet18_val.json")
    print(f"ResNet18 best (phase {phase}):", {k: round(v, 3) for k, v in best.items() if isinstance(v, float)})


def stage_compare(device, vl):
    rows = {
        "Majority-class dummy": json.load(open(RESULTS / "dummy_val.json")),
        "SimpleCNN (scratch)": json.load(open(RESULTS / "simplecnn_val.json")),
        "ResNet18 (fine-tuned)": json.load(open(RESULTS / "resnet18_val.json")),
    }
    cols = ["auc", "recall", "specificity", "precision", "f1", "accuracy"]
    cmp = pd.DataFrame(rows).T[cols].astype(float).round(3)
    cmp.to_csv(RESULTS / "comparison_val.csv")
    md = "| Model | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1) + "\n"
    for name, r in cmp.iterrows():
        md += f"| {name} | " + " | ".join(f"{v:.3f}" for v in r) + " |\n"
    (RESULTS / "comparison_val.md").write_text(md)
    print("\n" + md)

    simple = SimpleCNN().to(device)
    simple.load_state_dict(torch.load(CKPT / "simplecnn.pt", map_location=device))
    res = build_resnet18(freeze_backbone=False).to(device)
    res.load_state_dict(torch.load(CKPT / "resnet18_best.pt", map_location=device))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, mdl in [("SimpleCNN", simple), ("ResNet18", res)]:
        y, p, _ = predict(mdl, vl, device)
        RocCurveDisplay.from_predictions(y, p, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Majority class (AUC 0.5)")
    ax.legend(loc="lower right"); ax.set_title("Validation ROC")
    fig.savefig(RESULTS / "roc_val.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    y, p, _ = predict(res, vl, device)
    disp = ConfusionMatrixDisplay.from_predictions(
        y, (p >= 0.5).astype(int), display_labels=["Normal", "Pneumonia"], cmap="Blues")
    disp.ax_.set_title("ResNet18 validation confusion matrix")
    disp.figure_.savefig(RESULTS / "resnet_confusion_val.png", dpi=200, bbox_inches="tight")
    plt.close(disp.figure_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simple-epochs", type=int, default=12)
    ap.add_argument("--p1-epochs", type=int, default=4)
    ap.add_argument("--p2-epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fast", action="store_true", help="1 epoch per stage, to test the pipeline")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.fast:
        a.simple_epochs = a.p1_epochs = a.p2_epochs = 1

    RESULTS.mkdir(exist_ok=True)
    device = get_device()
    seed_everything(42)
    tl, vl, _ = get_loaders(bs=a.bs, workers=a.workers)   # test loader is never iterated
    criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weight().to(device))
    print("device:", device, "| train", len(tl.dataset), "| val", len(vl.dataset), flush=True)

    stage_dummy()
    stage_simplecnn(a, device, tl, vl, criterion)
    stage_resnet(a, device, tl, vl, criterion)
    stage_compare(device, vl)
    print("\nALL DONE. Results are in results/")


if __name__ == "__main__":
    main()
