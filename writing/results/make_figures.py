#!/usr/bin/env python
"""Generate publication figures (PNG 300 dpi) from experiment metrics for the paper.

Models compared: YOLOv8s (baseline), YOLO11s, YOLOv8s+CBAM.
Per-class mAP values (helmet, license_plate, motorcyclist) are from a manual
eval run of the YOLOv8s baseline (seed 42) on the test split. They are
hardcoded here because the dataset is not retained after the initial eval run.
"""
from __future__ import annotations
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

CV_DIR = Path(__file__).resolve().parents[2]
EXP = CV_DIR / "experiments"
OUT = CV_DIR / "writing" / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Colour palette — Okabe-Ito colour-blind safe
C = {
    "yolov8s": "#0072B2",
    "yolo11s":  "#009E73",
    "cbam":     "#CC79A7",
}

RUNS = {
    "YOLOv8s":      ["helm_yolov8s_roboflow_20260611", "ms_yolov8s_seed0", "ms_yolov8s_seed1"],
    "YOLO11s":      ["helm_yolo11s_roboflow_20260611",  "ms_yolo11s_seed0",  "ms_yolo11s_seed1"],
    "YOLOv8s+CBAM": ["ms_yolov8s_cbam_seed42",          "ms_yolov8s_cbam_seed0", "ms_yolov8s_cbam_seed1"],
}
COL = {
    "YOLOv8s":      C["yolov8s"],
    "YOLO11s":      C["yolo11s"],
    "YOLOv8s+CBAM": C["cbam"],
}
# FPS from a single dedicated run (GPU idle, batch=1) — seed 42 for each model
FPS_SINGLE = {
    "YOLOv8s":      296,
    "YOLO11s":      189,
    "YOLOv8s+CBAM": 290,
}


def load(metric: str) -> dict:
    """Load mean and std of a metric across seeds for each model."""
    out = {}
    for name, runs in RUNS.items():
        vals = []
        for r in runs:
            p = EXP / r / "metrics.json"
            if p.exists():
                vals.append(json.loads(p.read_text())[metric])
        if vals:
            out[name] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
        else:
            out[name] = (0.0, 0.0)
    return out


# ---------------------------------------------------------------------------
# Fig 1 — mAP@0.5 by architecture (main result)
# ---------------------------------------------------------------------------
m = load("mAP50")
names = list(RUNS)
means = [m[n][0] for n in names]
errs  = [m[n][1] for n in names]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    names, means, yerr=errs, capsize=6,
    color=[COL[n] for n in names],
    edgecolor="black", linewidth=0.7,
    error_kw=dict(elinewidth=1.2, ecolor="black"),
)
ax.set_ylim(0.93, 0.965)
ax.set_ylabel("mAP@0.5 (test split)", fontsize=10)
ax.set_title("Effect of attention mechanism on mAP@0.5", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
for b, v in zip(bars, means):
    ax.text(
        b.get_x() + b.get_width() / 2, v + 0.0008,
        f"{v:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
    )
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig1_attention_mAP50.png", dpi=300)
plt.close()
print("Saved fig1_attention_mAP50.png")


# ---------------------------------------------------------------------------
# Fig 2 — Per-class mAP@0.5 for YOLOv8s baseline (seed 42, test split)
# These values were obtained from the eval run of best.pt on the test split.
# ---------------------------------------------------------------------------
per_class = {
    "helmet":       0.923,
    "license_plate": 0.969,
    "motorcyclist": 0.990,
}
fig, ax = plt.subplots(figsize=(5.5, 4.4))
colors_pc = ["#0072B2", "#56B4E9", "#009E73"]
b = ax.bar(
    list(per_class), list(per_class.values()),
    color=colors_pc, edgecolor="black", linewidth=0.7
)
ax.set_ylim(0.87, 1.0)
ax.set_ylabel("mAP@0.5", fontsize=10)
ax.set_title("Per-class performance — YOLOv8s baseline (test split)", fontsize=10)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
for bb, v in zip(b, per_class.values()):
    ax.text(
        bb.get_x() + bb.get_width() / 2, v + 0.002,
        f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold"
    )
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.xticks(rotation=15, ha="right", fontsize=9)
plt.subplots_adjust(left=0.12, right=0.97, top=0.90, bottom=0.14)
plt.savefig(OUT / "fig2_per_class.png", dpi=300)
plt.close()
print("Saved fig2_per_class.png")


# ---------------------------------------------------------------------------
# Fig 3 — Accuracy vs speed trade-off
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4.2))
for n in names:
    ax.scatter(
        FPS_SINGLE[n], m[n][0],
        s=160, color=COL[n], edgecolor="black", linewidth=0.8,
        zorder=3, label=n
    )
    offset_x = 8 if n != "YOLO11s" else -80
    offset_y = 6 if n != "YOLOv8s+CBAM" else -14
    ax.annotate(
        n,
        (FPS_SINGLE[n], m[n][0]),
        textcoords="offset points",
        xytext=(offset_x, offset_y),
        fontsize=8.5,
    )

ax.set_xlabel("Inference speed (FPS, higher is better)", fontsize=10)
ax.set_ylabel("mAP@0.5 (test split)", fontsize=10)
ax.set_title("Accuracy vs. speed trade-off", fontsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
ax.grid(alpha=0.3, linestyle="--")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(100, 380)
ax.set_ylim(0.943, 0.964)
plt.tight_layout()
plt.savefig(OUT / "fig3_accuracy_speed.png", dpi=300)
plt.close()
print("Saved fig3_accuracy_speed.png")

print("\nAll figures saved to:", OUT)
for f in sorted(OUT.glob("*.png")):
    print(f"  - {f.name}  ({f.stat().st_size // 1024} KB)")
