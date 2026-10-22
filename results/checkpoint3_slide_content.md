# Checkpoint 3 slide content (auto-generated from results/)

## Slide 2: Timeline adherence
Week 1 data + preprocessing | Week 2 baseline + ResNet + diagrams | Week 3 tuning, test evaluation, Grad-CAM | Week 4 error analysis, README, demo. All complete.

## Slide 3: Test results (held-out, evaluated once)
Threshold 0.242 (validation-chosen, 95% recall target)

| model | auc | pr_auc | recall | specificity | precision | f1 | balanced_accuracy |
|---|---|---|---|---|---|---|---|
| Majority-class dummy | 0.500 [0.500, 0.500] | 0.625 [0.588, 0.663] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 0.625 [0.588, 0.663] | 0.769 [0.741, 0.798] | 0.500 [0.500, 0.500] |
| SimpleCNN @0.5 | 0.939 [0.917, 0.959] | 0.945 [0.914, 0.972] | 0.810 [0.770, 0.849] | 0.932 [0.896, 0.961] | 0.952 [0.925, 0.973] | 0.875 [0.848, 0.899] | 0.871 [0.844, 0.895] |
| SimpleCNN @val-threshold (0.038) | 0.939 [0.917, 0.959] | 0.945 [0.914, 0.972] | 0.990 [0.980, 0.998] | 0.560 [0.498, 0.625] | 0.789 [0.756, 0.827] | 0.878 [0.856, 0.902] | 0.775 [0.746, 0.808] |
| ResNet18 @0.5 | 0.981 [0.971, 0.990] | 0.986 [0.974, 0.994] | 0.990 [0.978, 0.998] | 0.821 [0.771, 0.868] | 0.902 [0.875, 0.931] | 0.944 [0.928, 0.960] | 0.905 [0.880, 0.929] |
| ResNet18 @val-threshold (0.242) | 0.981 [0.971, 0.990] | 0.986 [0.974, 0.994] | 0.995 [0.987, 1.000] | 0.786 [0.732, 0.837] | 0.886 [0.858, 0.917] | 0.937 [0.921, 0.954] | 0.891 [0.862, 0.917] |

## Slide 4: Versus baseline
Paired bootstrap AUC difference (ResNet18 minus SimpleCNN): +0.042 (95% CI +0.027 to +0.062, p = 0.001). McNemar exact test at each model's validation-chosen threshold: only ResNet18 correct = 62, only SimpleCNN correct = 7, p = 4.1e-12.

## Slide 5: Grad-CAM
Use results/gradcam_grid.png. Say what the heatmaps focus on, and discuss FP/FN.
Edge-mass table:

| group | count | mean | std |
|---|---|---|---|
| FN | 2 | 0.285 | 0.281 |
| FP | 50 | 0.375 | 0.09 |
| TN | 60 | 0.677 | 0.289 |
| TP | 60 | 0.372 | 0.043 |

## Slide 6: Live demo
streamlit run app.py (have a screen recording as backup)

## Slide 7: Limitations / future work
Pediatric single-center data; train/test distribution shift; possible train/test ID overlap; research prototype only. Future: external validation, adult data, calibration, multi-class.
