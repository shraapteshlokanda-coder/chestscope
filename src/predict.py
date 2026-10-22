"""Single-image inference: probability, label, confidence, Grad-CAM overlay.

    python -m src.predict path/to/xray.jpeg
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .gradcam import GradCAM, explain_pil, load_final_model, load_threshold
from .utils import ROOT


def load_predictor(device="cpu"):
    model = load_final_model(device)
    return model, GradCAM(model, model.layer4), load_threshold()


def predict_pil(model, cam_obj, threshold, pil_img, device="cpu"):
    out = explain_pil(model, cam_obj, pil_img, device)
    is_pna = out["prob"] >= threshold
    out["label"] = "Pneumonia likely" if is_pna else "No pneumonia flagged"
    out["confidence"] = out["prob"] if is_pna else 1 - out["prob"]
    out["threshold"] = threshold
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.predict path/to/xray.jpeg")
    model, cam_obj, th = load_predictor()
    out = predict_pil(model, cam_obj, th, Image.open(sys.argv[1]))
    print(f"{out['label']} | P(pneumonia)={out['prob']:.3f} | threshold={th:.3f}")
    print("Research prototype, not a diagnostic device.")
    dest = ROOT / "results" / "predictions"; dest.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
    ax[0].imshow(out["gray"], cmap="gray"); ax[0].set_title("Input"); ax[1].imshow(out["overlay"])
    ax[1].set_title(f"{out['label']} (P={out['prob']:.2f})")
    for a_ in ax: a_.axis("off")
    f = dest / (sys.argv[1].split("/")[-1].rsplit(".", 1)[0] + "_gradcam.png")
    fig.savefig(f, dpi=150, bbox_inches="tight"); print("Saved", f)