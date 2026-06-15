#!/usr/bin/env python
"""Entrypoint evaluasi — ukur metrik checkpoint di split test.

Contoh:
  python scripts/eval.py --config configs/baseline_yolov8.yaml \
      --checkpoint experiments/helm_yolov8s_roboflow_20260611/weights/best.pt

Metrik (mAP@0.5, mAP@[.5:.95], precision, recall, FPS) ditulis ke folder run.
Evaluasi memakai split TEST (CLAUDE.md: pastikan split yang benar, hindari leakage).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import print_env
from src.metrics import metrics_from_results, save_metrics
from src.seeding import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluasi YOLOv8 di split test")
    parser.add_argument("--config", required=True, help="Path config YAML")
    parser.add_argument("--checkpoint", required=True, help="Path bobot (best.pt)")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42), cfg.get("deterministic", True))
    print_env()

    from ultralytics import YOLO

    model = YOLO(args.checkpoint)
    # Auto-batch (-1) hanya untuk training; pakai batch tetap saat evaluasi.
    eval_batch = cfg["batch"] if cfg["batch"] and cfg["batch"] > 0 else 16
    print(f"\n▶ Evaluasi {args.checkpoint} di split '{args.split}' (batch={eval_batch})\n")
    results = model.val(
        data=cfg["data"],
        imgsz=cfg["imgsz"],
        batch=eval_batch,
        split=args.split,
        device=cfg.get("device", 0),
    )

    # FPS dari profil kecepatan inferensi (ms/gambar -> fps)
    fps = None
    speed = getattr(results, "speed", None)
    if speed and speed.get("inference"):
        fps = 1000.0 / speed["inference"]

    metrics = metrics_from_results(results, fps=fps)
    save_metrics(metrics, Path(args.checkpoint).resolve().parents[1])


if __name__ == "__main__":
    main()
