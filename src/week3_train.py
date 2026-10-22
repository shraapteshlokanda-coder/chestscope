"""Week 3: small hyperparameter comparison (validation only), seed variability,
final-model selection, and validation-chosen decision thresholds.

    python -m src.week3_train              # full (about 30 min on Apple Silicon)
    python -m src.week3_train --skip-seeds # tuning only (about 15 min)

Resumable: finished runs (results/tune/*.json) are skipped.
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import json
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import get_loaders, get_pos_weight
from .evaluate import predict
from .models import SimpleCNN, build_resnet18, unfreeze_all, unfreeze_layers
from .stats_utils import all_metrics, pick_threshold
from .train import fit
from .utils import ROOT, get_device, save_json, seed_everything

RESULTS = ROOT / "results"
CKPT = RESULTS / "checkpoints"
TUNE = RESULTS / "tune"

CONFIGS = {
    "base": dict(lr_backbone=1e-4, unfreeze="all"),
    "lr3e-4": dict(lr_backbone=3e-4, unfreeze="all"),
    "layer34": dict(lr_backbone=1e-4, unfreeze="layer34"),
}


def run_config(name, seed, cfg, device, tl, vl, criterion, p1, p2):
    tag = f"{name}_s{seed}"
    out = TUNE / f"{tag}.json"
    if out.exists():
        print(f"{tag}: already done, skipping")
        return json.load(open(out))
    print(f"\n=== {tag}: {cfg} ===", flush=True)
    seed_everything(seed)
    model = build_resnet18(freeze_backbone=True).to(device)
    opt = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
    _, b1 = fit(model, tl, vl, criterion, opt, device, p1, CKPT / f"{tag}_p1.pt")

    if cfg["unfreeze"] == "all":
        unfreeze_all(model)
    else:
        unfreeze_layers(model, ("layer3", "layer4", "fc"))
    backbone = [p for n, p in model.named_parameters() if not n.startswith("fc") and p.requires_grad]
    opt = torch.optim.Adam([{"params": backbone, "lr": cfg["lr_backbone"]},
                            {"params": model.fc.parameters(), "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    _, b2 = fit(model, tl, vl, criterion, opt, device, p2, CKPT / f"{tag}_p2.pt", scheduler=sched)

    best, phase = (b2, 2) if b2["auc"] >= b1["auc"] else (b1, 1)
    shutil.copy(CKPT / f"{tag}_p{phase}.pt", CKPT / f"{tag}.pt")
    best = {**best, "phase": phase, "config": name, "seed": seed}
    save_json(best, out)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1", type=int, default=4)
    ap.add_argument("--p2", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip-seeds", action="store_true")
    ap.add_argument("--target-recall", type=float, default=0.95)
    a = ap.parse_args()

    TUNE.mkdir(parents=True, exist_ok=True)
    device = get_device()
    tl, vl, _ = get_loaders(bs=a.bs, workers=a.workers)          # test loader never iterated
    criterion = nn.BCEWithLogitsLoss(pos_weight=get_pos_weight().to(device))

    # --- reuse the Checkpoint 2 run as the "base" config, seed 42
    if not (TUNE / "base_s42.json").exists() and (RESULTS / "resnet18_val.json").exists():
        shutil.copy(CKPT / "resnet18_best.pt", CKPT / "base_s42.pt")
        b = json.load(open(RESULTS / "resnet18_val.json"))
        save_json({**b, "config": "base", "seed": 42}, TUNE / "base_s42.json")
        print("Reused Checkpoint 2 ResNet18 run as base_s42")

    # --- 1. hyperparameter comparison (seed 42)
    rows = []
    for name, cfg in CONFIGS.items():
        r = run_config(name, 42, cfg, device, tl, vl, criterion, a.p1, a.p2)
        rows.append(dict(config=name, **cfg, val_auc=r["auc"], val_recall=r["recall"],
                         val_specificity=r["specificity"], val_f1=r["f1"], val_loss=r["loss"]))
    tuning = pd.DataFrame(rows)
    tuning.to_csv(RESULTS / "tuning.csv", index=False)
    print("\n", tuning.round(4).to_string(index=False))

    # best by val AUC (rounded to 3 dp), ties broken by lower val loss
    ranked = tuning.assign(auc3=tuning.val_auc.round(3)).sort_values(["auc3", "val_loss"], ascending=[False, True])
    best_name = ranked.iloc[0].config
    print("Best config on validation:", best_name)

    # --- 2. seed variability for the best config
    if not a.skip_seeds:
        seed_rows = []
        for seed in (42, 1, 2):
            r = run_config(best_name, seed, CONFIGS[best_name], device, tl, vl, criterion, a.p1, a.p2)
            seed_rows.append(dict(seed=seed, val_auc=r["auc"], val_recall=r["recall"],
                                  val_specificity=r["specificity"], val_f1=r["f1"]))
        sd = pd.DataFrame(seed_rows)
        summ = pd.DataFrame({"mean": sd.drop(columns="seed").mean(), "std": sd.drop(columns="seed").std()}).T
        pd.concat([sd, summ.reset_index().rename(columns={"index": "seed"})]).to_csv(RESULTS / "seeds.csv", index=False)
        print("\nSeed variability:\n", sd.round(4).to_string(index=False))

    # --- 3. final model = best config, seed 42 (selected on validation only)
    shutil.copy(CKPT / f"{best_name}_s42.pt", CKPT / "final.pt")
    save_json({"config": best_name, **CONFIGS[best_name], "seed": 42,
               "selected_on": "validation AUC, ties by validation loss"}, RESULTS / "final_config.json")

    # --- 4. decision thresholds chosen on VALIDATION (target recall)
    final = build_resnet18(freeze_backbone=False, pretrained=False).to(device)
    final.load_state_dict(torch.load(CKPT / "final.pt", map_location=device))
    simple = SimpleCNN().to(device)
    simple.load_state_dict(torch.load(CKPT / "simplecnn.pt", map_location=device))
    th = {"target_recall": a.target_recall, "chosen_on": "validation set"}
    for name, mdl in [("resnet18", final), ("simplecnn", simple)]:
        y, p, _ = predict(mdl, vl, device)
        t = pick_threshold(y, p, a.target_recall)
        m = all_metrics(y, p, t)
        th[name] = {"threshold": t, **{k: round(v, 4) for k, v in m.items() if isinstance(v, float)}}
        print(f"{name}: threshold {t:.4f} -> val recall {m['recall']:.3f}, specificity {m['specificity']:.3f}")
    save_json(th, RESULTS / "threshold.json")
    print("\nWEEK 3 TRAINING DONE. Final model: results/checkpoints/final.pt, thresholds: results/threshold.json")


if __name__ == "__main__":
    main()