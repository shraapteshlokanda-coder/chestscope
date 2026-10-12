import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score


def compute_metrics(y, p, thresh=0.5):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thresh).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        auc=float(roc_auc_score(y, p)),
        precision=float(pr), recall=float(rc), f1=float(f1),
        specificity=float(tn / (tn + fp)) if (tn + fp) else 0.0,
        accuracy=float((pred == y).mean()),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


@torch.no_grad()
def predict(model, loader, device, criterion=None):
    model.eval()
    probs, ys, total = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)
        if criterion is not None:
            total += criterion(logits, y).item() * x.size(0)
        probs.append(torch.sigmoid(logits).cpu())
        ys.append(y.cpu())
    loss = total / len(loader.dataset) if criterion is not None else None
    return torch.cat(ys).numpy(), torch.cat(probs).numpy(), loss


def evaluate(model, loader, device, criterion=None, thresh=0.5):
    y, p, loss = predict(model, loader, device, criterion)
    m = compute_metrics(y, p, thresh)
    if loss is not None:
        m["loss"] = float(loss)
    return m
