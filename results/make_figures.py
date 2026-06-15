#!/usr/bin/env python
"""Hasilkan figur publikasi (PNG 300 dpi) dari metrik eksperimen untuk paper."""
from __future__ import annotations
import json, shutil
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
EXP = REPO / "experiments"
OUT = REPO / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palet ramah buta-warna (Okabe-Ito)
C = {"yolov8s": "#0072B2", "yolo11s": "#009E73", "rtdetr": "#D55E00", "cbam": "#CC79A7"}

RUNS = {
    "YOLOv8s":      ["helm_yolov8s_roboflow_20260611", "ms_yolov8s_seed0", "ms_yolov8s_seed1"],
    "YOLO11s":      ["helm_yolo11s_roboflow_20260611", "ms_yolo11s_seed0", "ms_yolo11s_seed1"],
    "RT-DETR-l":    ["ms_rtdetr_seed42", "ms_rtdetr_seed0"],
    "YOLOv8s+CBAM": ["ms_yolov8s_cbam_seed42", "ms_yolov8s_cbam_seed0", "ms_yolov8s_cbam_seed1"],
}
COL = {"YOLOv8s": C["yolov8s"], "YOLO11s": C["yolo11s"], "RT-DETR-l": C["rtdetr"], "YOLOv8s+CBAM": C["cbam"]}
# FPS andal dari run tunggal (seed 42) saat GPU senggang
FPS_SINGLE = {"YOLOv8s": 296, "YOLO11s": 189, "RT-DETR-l": 55, "YOLOv8s+CBAM": 290}


def load(metric):
    out = {}
    for name, runs in RUNS.items():
        vals = []
        for r in runs:
            p = EXP / r / "metrics.json"
            if p.exists():
                vals.append(json.loads(p.read_text())[metric])
        out[name] = (st.mean(vals), st.stdev(vals) if len(vals) > 1 else 0.0)
    return out


# --- Fig 1: spektrum atensi vs mAP@0.5 (hasil utama) ---
m = load("mAP50")
names = list(RUNS); means = [m[n][0] for n in names]; errs = [m[n][1] for n in names]
fig, ax = plt.subplots(figsize=(7, 4.2))
bars = ax.bar(names, means, yerr=errs, capsize=5, color=[COL[n] for n in names], edgecolor="black", linewidth=0.6)
ax.set_ylim(0.92, 0.97); ax.set_ylabel("mAP@0.5 (split uji)")
ax.set_title("Pengaruh mekanisme atensi terhadap mAP@0.5")
for b, v in zip(bars, means):
    ax.text(b.get_x()+b.get_width()/2, v+0.0012, f"{v:.4f}", ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.3); plt.xticks(rotation=12)
plt.tight_layout(); plt.savefig(OUT/"fig1_attention_mAP50.png", dpi=300); plt.close()

# --- Fig 2: mAP@0.5 per-kelas (baseline YOLOv8s) ---
per_class = {"helmet": 0.923, "license_plate": 0.969, "motorcyclist": 0.990}
fig, ax = plt.subplots(figsize=(6, 4))
b = ax.bar(list(per_class), list(per_class.values()),
           color=["#0072B2", "#56B4E9", "#009E73"], edgecolor="black", linewidth=0.6)
ax.set_ylim(0.85, 1.0); ax.set_ylabel("mAP@0.5"); ax.set_title("Performa per-kelas (YOLOv8s, split uji)")
for bb, v in zip(b, per_class.values()):
    ax.text(bb.get_x()+bb.get_width()/2, v+0.002, f"{v:.3f}", ha="center", fontsize=9)
ax.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(OUT/"fig2_per_class.png", dpi=300); plt.close()

# --- Fig 3: akurasi vs kecepatan (mAP@0.5 vs FPS) ---
fig, ax = plt.subplots(figsize=(6.5, 4.4))
for n in names:
    ax.scatter(FPS_SINGLE[n], m[n][0], s=140, color=COL[n], edgecolor="black", zorder=3, label=n)
    ax.annotate(n, (FPS_SINGLE[n], m[n][0]), textcoords="offset points", xytext=(8, 6), fontsize=8)
ax.set_xlabel("Kecepatan inferensi (FPS, makin kanan makin cepat)")
ax.set_ylabel("mAP@0.5"); ax.set_title("Trade-off akurasi vs kecepatan")
ax.grid(alpha=0.3); ax.set_ylim(0.94, 0.965)
plt.tight_layout(); plt.savefig(OUT/"fig3_accuracy_speed.png", dpi=300); plt.close()

# --- Fig 4: contoh deteksi kualitatif ---
src = sorted((REPO/"results"/"predict").glob("*.jpg"))
if src:
    shutil.copy(src[0], OUT/"fig4_detection_example.png")

print("✓ Figur tersimpan di", OUT)
for f in sorted(OUT.glob("*.png")):
    print("  -", f.name)
