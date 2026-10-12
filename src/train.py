import time
from pathlib import Path

import pandas as pd
import torch

from .evaluate import evaluate


def train_one_epoch(model, loader, opt, criterion, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = criterion(model(x).squeeze(1), y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


def fit(model, train_loader, val_loader, criterion, opt, device, epochs,
        ckpt_path, scheduler=None):
    """Train; save best checkpoint by val ROC-AUC; reload it at the end.
    Returns (history DataFrame, best-epoch val metrics dict). Never touches the test set."""
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    rows, best_auc, best = [], -1.0, None

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, opt, criterion, device)
        m = evaluate(model, val_loader, device, criterion=criterion)
        if scheduler is not None:
            scheduler.step(m["auc"])
        secs = time.time() - t0
        rows.append(dict(
            epoch=ep, train_loss=tr_loss, val_loss=m["loss"], val_auc=m["auc"],
            val_recall=m["recall"], val_specificity=m["specificity"],
            val_precision=m["precision"], val_f1=m["f1"],
            lr=opt.param_groups[-1]["lr"], secs=secs,
        ))
        print(f"ep {ep:02d}/{epochs} | train {tr_loss:.3f} | val {m['loss']:.3f} | AUC {m['auc']:.3f} | "
              f"rec {m['recall']:.3f} | spec {m['specificity']:.3f} | {secs:.0f}s", flush=True)
        if m["auc"] > best_auc:
            best_auc, best = m["auc"], {**m, "epoch": ep}
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return pd.DataFrame(rows), best
