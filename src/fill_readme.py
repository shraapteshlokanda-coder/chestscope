"""Inserts Week 3-4 results into README.md and writes results/checkpoint3_slide_content.md.

    python -m src.fill_readme
"""
import json

import pandas as pd

from .utils import ROOT

R = ROOT / "results"


def md_table(df):
    cols = list(df.columns)
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(str(r[c]) for c in cols) + " |\n"
    return s


def read(name, default=""):
    f = R / name
    return f.read_text() if f.exists() else default


def main():
    th = json.load(open(R / "threshold.json"))
    stats = json.load(open(R / "test_stats.json"))
    test_md = read("test_metrics.md")
    tuning = md_table(pd.read_csv(R / "tuning.csv").round(4)) if (R / "tuning.csv").exists() else ""
    seeds = md_table(pd.read_csv(R / "seeds.csv").round(4)) if (R / "seeds.csv").exists() else "(seed runs skipped)\n"
    edge = md_table(pd.read_csv(R / "gradcam_edge_mass.csv").round(3)) if (R / "gradcam_edge_mass.csv").exists() else ""
    d, mc = stats["auc_diff_resnet_minus_simplecnn"], stats["mcnemar_at_val_thresholds"]
    t_res = th["resnet18"]["threshold"]

    stat_line = (f"Paired bootstrap AUC difference (ResNet18 minus SimpleCNN): {d['diff']:+.3f} "
                 f"(95% CI {d['lo']:+.3f} to {d['hi']:+.3f}, p = {d['p_value']:.3f}). "
                 f"McNemar exact test at each model's validation-chosen threshold: only ResNet18 correct = "
                 f"{mc['only_a_correct']}, only SimpleCNN correct = {mc['only_b_correct']}, p = {mc['p_value']:.3g}.")

    section = f"""## Test-set results (held-out; evaluated once)
Decision threshold for ResNet18 = {t_res:.3f}, chosen on the **validation** set to reach {th['target_recall']:.0%} recall.
Values are point estimate [95% bootstrap CI, 1,000 resamples].

{test_md}
**Statistical comparison against the baseline.** {stat_line}

![Test ROC](results/roc_test.png)
![Test confusion matrices](results/test_confusion.png)

### Hyperparameter comparison (validation set)
{tuning}
### Seed variability of the selected configuration (validation set)
{seeds}
## Explainability (Grad-CAM)
![Grad-CAM grid](results/gradcam_grid.png)

Share of Grad-CAM heat falling on the outer image border (possible shortcut indicator), by outcome group:

{edge}
**Error analysis:** <!-- WRITE 3-4 observations after viewing results/gradcam_grid.png: do heatmaps sit on lung fields or on borders/markers? what do FP/FN have in common? -->

## Demo
```bash
pip install streamlit
streamlit run app.py        # upload an X-ray: prediction, confidence, Grad-CAM overlay
python -m src.predict path/to/xray.jpeg
```
Model weights (`results/checkpoints/final.pt`) are not committed; <!-- ADD link to GitHub Release asset -->.

"""
    readme = (ROOT / "README.md").read_text()
    if "## Test-set results" not in readme:
        readme = readme.replace("## Limitations", section + "## Limitations", 1)
        (ROOT / "README.md").write_text(readme)
        print("README.md updated.")
    else:
        print("README already has test results; skipped.")

    # ---- slide content sheet
    lines = [
        "# Checkpoint 3 slide content (auto-generated from results/)\n",
        "## Slide 2: Timeline adherence\nWeek 1 data + preprocessing | Week 2 baseline + ResNet + diagrams | "
        "Week 3 tuning, test evaluation, Grad-CAM | Week 4 error analysis, README, demo. All complete.\n",
        f"## Slide 3: Test results (held-out, evaluated once)\nThreshold {t_res:.3f} (validation-chosen, {th['target_recall']:.0%} recall target)\n\n{test_md}",
        f"## Slide 4: Versus baseline\n{stat_line}\n",
        "## Slide 5: Grad-CAM\nUse results/gradcam_grid.png. Say what the heatmaps focus on, and discuss FP/FN.\n"
        f"Edge-mass table:\n\n{edge}",
        "## Slide 6: Live demo\nstreamlit run app.py (have a screen recording as backup)\n",
        "## Slide 7: Limitations / future work\nPediatric single-center data; train/test distribution shift; possible train/test ID overlap; "
        "research prototype only. Future: external validation, adult data, calibration, multi-class.\n",
    ]
    (R / "checkpoint3_slide_content.md").write_text("\n".join(lines))
    print("Wrote results/checkpoint3_slide_content.md")


if __name__ == "__main__":
    main()