"""THE one-time held-out TEST evaluation. Run only after the model and thresholds are frozen.

    python -m src.test_eval

Refuses to run twice (results/test_results.json exists) unless --force is given.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .data import SPLITS, get_loaders
from .evaluate import predict
from .models import SimpleCNN, build_resnet18
from .stats_utils import METRICS, all_metrics, bootstrap_ci, mcnemar_exact, paired_bootstrap_auc_diff
from .utils import ROOT, get_device, save_json

RESULTS = ROOT / "results"
CKPT = RESULTS / "checkpoints"


def fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    if (RESULTS / "test_results.json").exists() and not a.force:
        print("Test set was already evaluated (results/test_results.json). The test set is used ONCE.\n"
              "Re-run with --force only for a bug fix, and do not change the model or threshold after seeing results.")
        return

    device = get_device()
    th = json.load(open(RESULTS / "threshold.json"))
    t_res, t_sim = th["resnet18"]["threshold"], th["simplecnn"]["threshold"]

    res = build_resnet18(freeze_backbone=False, pretrained=False).to(device)
    res.load_state_dict(torch.load(CKPT / "final.pt", map_location=device))
    sim = SimpleCNN().to(device)
    sim.load_state_dict(torch.load(CKPT / "simplecnn.pt", map_location=device))

    _, _, test_loader = get_loaders(bs=32, workers=0)
    y, p_res, _ = predict(res, test_loader, device)
    _, p_sim, _ = predict(sim, test_loader, device)
    p_dummy = np.ones(len(y))
    print(f"Test set: {len(y)} images, {y.mean()*100:.1f}% pneumonia")

    ops = [("Majority-class dummy", p_dummy, 0.5),
           ("SimpleCNN @0.5", p_sim, 0.5),
           (f"SimpleCNN @val-threshold ({t_sim:.3f})", p_sim, t_sim),
           ("ResNet18 @0.5", p_res, 0.5),
           (f"ResNet18 @val-threshold ({t_res:.3f})", p_res, t_res)]
    results, rows = {}, []
    for name, p, t in ops:
        ci, point = bootstrap_ci(y, p, t, n=a.n_boot)
        results[name] = {"threshold": t, "metrics": {k: list(v) for k, v in ci.items()},
                         "counts": {k: point[k] for k in ("tp", "fp", "tn", "fn")}}
        rows.append({"model": name, **{m: fmt(ci[m]) for m in METRICS}})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "test_metrics.csv", index=False)
    cols = list(table.columns)
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in table.iterrows():
        md += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    (RESULTS / "test_metrics.md").write_text(md)
    print(md)

    # statistical comparison: ResNet18 vs SimpleCNN (both at their validation-chosen thresholds)
    stats = {
        "auc_diff_resnet_minus_simplecnn": paired_bootstrap_auc_diff(y, p_res, p_sim),
        "mcnemar_at_val_thresholds": mcnemar_exact(y, p_res >= t_res, p_sim >= t_sim),
        "mcnemar_at_0.5": mcnemar_exact(y, p_res >= 0.5, p_sim >= 0.5),
    }
    save_json(stats, RESULTS / "test_stats.json")
    print(json.dumps(stats, indent=2))
    save_json({"n_test": int(len(y)), "pneumonia_pct": float(y.mean() * 100), "operating_points": results},
              RESULTS / "test_results.json")

    # per-image predictions (used for Grad-CAM error analysis)
    paths = pd.read_csv(SPLITS / "test.csv")["path"]
    pd.DataFrame({"path": paths, "y": y.astype(int), "p_resnet": p_res, "p_simplecnn": p_sim,
                  "pred_resnet": (p_res >= t_res).astype(int)}).to_csv(RESULTS / "test_predictions.csv", index=False)

    # figures
    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y, p_res, name="ResNet18", ax=ax)
    RocCurveDisplay.from_predictions(y, p_sim, name="SimpleCNN", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Majority class (AUC 0.5)")
    ax.legend(loc="lower right"); ax.set_title("Held-out TEST ROC")
    fig.savefig(RESULTS / "roc_test.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, t, title in [(axes[0], 0.5, "ResNet18 @ 0.5"), (axes[1], t_res, f"ResNet18 @ val-threshold {t_res:.3f}")]:
        ConfusionMatrixDisplay.from_predictions(y, (p_res >= t).astype(int), display_labels=["Normal", "Pneumonia"],
                                                cmap="Blues", ax=ax, colorbar=False)
        ax.set_title(title)
    fig.savefig(RESULTS / "test_confusion.png", dpi=200, bbox_inches="tight"); plt.close(fig)
    print("\nDONE. Saved test_metrics.md/csv, test_stats.json, test_predictions.csv, roc_test.png, test_confusion.png")


if __name__ == "__main__":
    main()