"""Grad-CAM for the ResNet18 (target: layer4, positive/pneumonia logit). Implemented from scratch
(forward/backward hooks) so there is no extra dependency.

    python -m src.gradcam      # builds results/gradcam_grid.png and results/gradcam_edge_mass.csv
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .data import eval_tf
from .models import build_resnet18
from .utils import ROOT

RESULTS = ROOT / "results"


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output
        output.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x):
        """x: [1,3,224,224] normalized tensor. Returns (cam in [0,1] as HxW numpy, P(pneumonia))."""
        self.model.zero_grad()
        logit = self.model(x).squeeze()
        logit.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, float(torch.sigmoid(logit).item())


def load_final_model(device="cpu"):
    m = build_resnet18(freeze_backbone=False, pretrained=False)
    m.load_state_dict(torch.load(RESULTS / "checkpoints" / "final.pt", map_location=device))
    for p in m.parameters():
        p.requires_grad = True
    return m.to(device).eval()


def load_threshold(default=0.5):
    f = RESULTS / "threshold.json"
    return json.load(open(f))["resnet18"]["threshold"] if f.exists() else default


def overlay(gray_224, cam, alpha=0.45):
    """gray_224: HxW uint8 array. Returns HxWx3 float overlay in [0,1]."""
    rgb = np.repeat((gray_224.astype(np.float32) / 255.0)[..., None], 3, axis=2)
    heat = plt.get_cmap("jet")(cam)[..., :3]
    return np.clip((1 - alpha) * rgb + alpha * heat, 0, 1)


def explain_pil(model, cam_obj, pil_img, device="cpu"):
    gray = pil_img.convert("L")
    x = eval_tf(gray).unsqueeze(0).to(device)
    cam, prob = cam_obj(x)
    disp = np.array(gray.resize((224, 224)))
    return dict(prob=prob, cam=cam, gray=disp, overlay=overlay(disp, cam))


def edge_mass(cam, border=34):
    """Fraction of heat outside the central region (outer ~15% border on each side)."""
    total = cam.sum() + 1e-8
    return float(1 - cam[border:224 - border, border:224 - border].sum() / total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--edge-n", type=int, default=60)
    a = ap.parse_args()

    df = pd.read_csv(RESULTS / "test_predictions.csv")
    th = load_threshold()
    df["pred"] = (df.p_resnet >= th).astype(int)
    df["group"] = np.select(
        [(df.y == 1) & (df.pred == 1), (df.y == 0) & (df.pred == 0), (df.y == 0) & (df.pred == 1)],
        ["TP", "TN", "FP"], default="FN")
    print(df.group.value_counts().to_dict(), "| threshold", round(th, 4))

    model = load_final_model("cpu")
    cam_obj = GradCAM(model, model.layer4)

    # --- grid: 4 groups x per-group examples (errors = most confident mistakes)
    rng = np.random.default_rng(0)
    picks = {}
    for g in ["TP", "TN", "FP", "FN"]:
        sub = df[df.group == g]
        if g in ("FP", "FN"):
            err = (sub.p_resnet - sub.y).abs()
            sub = sub.loc[err.sort_values(ascending=False).index]
            picks[g] = sub.head(a.per_group)
        else:
            picks[g] = sub.sample(min(a.per_group, len(sub)), random_state=int(rng.integers(1e6)))
    names = {"TP": "True positive", "TN": "True negative", "FP": "False positive", "FN": "False negative"}
    fig, axes = plt.subplots(4, a.per_group, figsize=(3 * a.per_group, 12.5))
    for r, g in enumerate(["TP", "TN", "FP", "FN"]):
        rows = list(picks[g].itertuples())
        for c in range(a.per_group):
            ax = axes[r, c]; ax.axis("off")
            if c < len(rows):
                out = explain_pil(model, cam_obj, Image.open(rows[c].path))
                ax.imshow(out["overlay"])
                ax.set_title(f"{names[g]}\ntrue={'PNA' if rows[c].y else 'NORM'}  p={out['prob']:.2f}", fontsize=9)
    fig.suptitle("Grad-CAM (ResNet18 layer4, pneumonia logit): held-out test set", fontsize=13)
    fig.savefig(RESULTS / "gradcam_grid.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # --- quantitative shortcut check: how much heat falls on the image border?
    recs = []
    for g in ["TP", "TN", "FP", "FN"]:
        sub = df[df.group == g]
        sub = sub.sample(min(a.edge_n, len(sub)), random_state=0)
        for p_ in sub.path:
            cam, _ = cam_obj(eval_tf(Image.open(p_).convert("L")).unsqueeze(0))
            recs.append(dict(group=g, edge_mass=edge_mass(cam)))
    em = pd.DataFrame(recs).groupby("group").edge_mass.agg(["count", "mean", "std"]).round(3)
    em.to_csv(RESULTS / "gradcam_edge_mass.csv")
    print(em)
    print("Saved results/gradcam_grid.png and results/gradcam_edge_mass.csv")


if __name__ == "__main__":
    main()