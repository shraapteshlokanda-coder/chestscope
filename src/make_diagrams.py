"""Draws the pipeline and architecture diagrams as PNGs in docs/diagrams/.

    python -m src.make_diagrams
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OUT = ROOT / "docs" / "diagrams"

COL = dict(data="#DCEBFA", proc="#E3F1E1", model="#FCE9C8", out="#F6D9D9",
           locked="#E4E4E4", cam="#EBDDF5", frozen="#D9E2EE")


def _box(ax, cx, cy, w, h, text, color, dashed=False, edge="#333", fs=9, bold=False):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                       fc=color, ec=edge, lw=1.6, ls="--" if dashed else "-")
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.3)
    return b


def _arrow(ax, a, b, dashed=False):
    ax.add_patch(FancyArrowPatch(a[0], b[0], patchA=a[1], patchB=b[1], arrowstyle="-|>",
                                 mutation_scale=16, lw=1.5, color="#333",
                                 ls="--" if dashed else "-", shrinkA=2, shrinkB=2))


def pipeline():
    fig, ax = plt.subplots(figsize=(18, 6.5))
    ax.set_xlim(0, 150); ax.set_ylim(-8, 60); ax.axis("off")
    W, H = 21, 13
    n = {}
    n["raw"] = (12, 32, "Kaggle Chest X-Ray\n(pediatric, 5,856 images)", COL["data"])
    n["pool"] = (36, 32, "Pool official\ntrain + val", COL["data"])
    n["split"] = (60, 32, "Patient-grouped split\n(StratifiedGroupKFold,\nseed 42)", COL["proc"])
    n["tr"] = (86, 46, "TRAIN preprocessing\ngray to 3ch, resize 256,\ncrop 224, rotate +/-10,\nbrightness/contrast, normalize", COL["proc"])
    n["va"] = (86, 18, "VAL preprocessing\ngray to 3ch, resize 224,\nnormalize (ImageNet)", COL["proc"])
    n["model"] = (112, 32, "ResNet18 backbone\n+ FC head (1 logit)\nweighted BCE loss", COL["model"])
    n["out"] = (138, 44, "Sigmoid:\nprediction +\nconfidence score", COL["out"])
    n["cam"] = (138, 20, "Grad-CAM heatmap\noverlay (Week 3)", COL["cam"])
    n["test"] = (12, 8, "Official TEST set\nLOCKED until Week 3", COL["locked"])
    n["final"] = (112, 8, "Final test evaluation\n(Week 3)", COL["locked"])
    B = {}
    for k, (x, y, t, c) in n.items():
        w = 27 if k in ("tr", "va") else W
        h = 15 if k == "tr" else H
        B[k] = (( x, y), _box(ax, x, y, w, h, t, c, dashed=k in ("cam", "test", "final"), fs=9))
    for a, b, d in [("raw", "pool", 0), ("pool", "split", 0), ("split", "tr", 0), ("split", "va", 0),
                    ("tr", "model", 0), ("va", "model", 0), ("model", "out", 0),
                    ("model", "cam", 1), ("test", "final", 1), ("raw", "test", 1), ("final", "model", 1)]:
        _arrow(ax, (B[a][0], B[a][1]), (B[b][0], B[b][1]), dashed=bool(d))
    ax.text(75, 58, "ChestScope: end-to-end data and model pipeline", ha="center", fontsize=15, fontweight="bold")
    ax.text(75, -6, "Solid = implemented (Weeks 1-2).  Dashed = Week 3 / locked.", ha="center", fontsize=10, style="italic")
    fig.savefig(OUT / "pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def chain(labels, colors, title, fname, notes, step=19, w=16, h=14):
    fig, ax = plt.subplots(figsize=(21, 5))
    ax.set_xlim(-w / 2 - 1, step * (len(labels) - 1) + w / 2 + 1); ax.set_ylim(-14, 24); ax.axis("off")
    boxes = []
    for i, (t, c) in enumerate(zip(labels, colors)):
        boxes.append(((i * step, 6), _box(ax, i * step, 6, w, h, t, c, fs=8.8, edge="#B22" if "Grad-CAM" in t else "#333")))
    for a, b in zip(boxes[:-1], boxes[1:]):
        _arrow(ax, a, b)
    ax.text(step * (len(labels) - 1) / 2, 22, title, ha="center", fontsize=15, fontweight="bold")
    ax.text(step * (len(labels) - 1) / 2, -9, notes, ha="center", fontsize=10.5, va="center", linespacing=1.5)
    fig.savefig(OUT / fname, dpi=200, bbox_inches="tight")
    plt.close(fig)


def architecture():
    labels = ["Input\n3 x 224 x 224", "Stem\n7x7 conv, BN, ReLU,\nmaxpool\n64 x 56 x 56",
              "layer1\n2 BasicBlocks\n64 x 56 x 56", "layer2\n2 BasicBlocks\n128 x 28 x 28",
              "layer3\n2 BasicBlocks\n256 x 14 x 14", "layer4\n2 BasicBlocks\n512 x 7 x 7\nGrad-CAM target",
              "Global avg pool\n512", "FC 512 to 1\n(new head)", "Sigmoid\nP(pneumonia)"]
    colors = [COL["data"]] + [COL["frozen"]] * 5 + [COL["model"]] * 2 + [COL["out"]]
    notes = ("ImageNet-pretrained ResNet18.   Phase 1: backbone frozen (blue-grey), only FC head trains, lr 1e-3.\n"
             "Phase 2: all layers unfrozen, backbone lr 1e-4, head lr 1e-3, ReduceLROnPlateau.   Loss: BCE with logits + pos_weight.")
    chain(labels, colors, "ChestScope model architecture: ResNet18 transfer learning", "architecture.png", notes)


def baseline():
    labels = ["Input\n3 x 224 x 224", "Block 1\nConv3x3 16\nBN, ReLU, MaxPool", "Block 2\nConv3x3 32\nBN, ReLU, MaxPool",
              "Block 3\nConv3x3 64\nBN, ReLU, MaxPool", "Block 4\nConv3x3 128\nBN, ReLU, MaxPool",
              "Global avg pool\n128", "Dropout 0.3", "FC 128 to 1", "Sigmoid\nP(pneumonia)"]
    colors = [COL["data"]] + [COL["proc"]] * 4 + [COL["model"]] * 3 + [COL["out"]]
    notes = "Baseline trained from scratch: Adam lr 1e-3, 12 epochs, weighted BCE loss, best checkpoint by validation ROC-AUC."
    chain(labels, colors, "Baseline model: SimpleCNN (from scratch)", "baseline_architecture.png", notes)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline(); architecture(); baseline()
    print("Diagrams saved to", OUT)
