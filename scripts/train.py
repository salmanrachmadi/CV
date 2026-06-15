#!/usr/bin/env python
"""Entrypoint training YOLOv8 — baca config, patok seed, catat env, latih.

Contoh:
  # Training penuh
  python scripts/train.py --config configs/baseline_yolov8.yaml

  # Dry-run cepat (subset kecil) sebelum training panjang (lihat CLAUDE.md)
  python scripts/train.py --config configs/baseline_yolov8.yaml --epochs 2 --subset 0.05

Output disimpan ke experiments/<run_name>_<tanggal>/ beserta config & env.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import print_env
from src.seeding import set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Latih baseline YOLOv8 deteksi helm")
    parser.add_argument("--config", required=True, help="Path config YAML")
    parser.add_argument("--epochs", type=int, help="Override epochs (mis. dry-run)")
    parser.add_argument(
        "--subset",
        type=float,
        help="Override fraction data latih (0-1) untuk dry-run cepat",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.subset is not None:
        cfg["fraction"] = args.subset  # Ultralytics: 'fraction'

    # Reproducibility
    seed = set_seed(cfg.get("seed", 42), cfg.get("deterministic", True))
    env = print_env()

    # Nama run: <run_name>_<tanggal> (tanggal eksplisit, bukan Date.now di kode)
    run_name = f"{cfg['run_name']}_{date.today():%Y%m%d}"
    project_dir = REPO_ROOT / cfg.get("project_dir", "experiments")

    from ultralytics import YOLO

    model = YOLO(cfg["model"])
    aug = cfg.get("augment", {})

    print(f"\n▶ Mulai training: {run_name}  (seed={seed}, device={cfg.get('device')})\n")
    model.train(
        data=cfg["data"],
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        optimizer=cfg.get("optimizer", "auto"),
        lr0=cfg.get("lr0", 0.01),
        patience=cfg.get("patience", 25),
        seed=seed,
        deterministic=cfg.get("deterministic", True),
        device=cfg.get("device", 0),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        fraction=cfg.get("fraction", 1.0),
        **aug,
    )

    # Catat config + env ke folder run (reproducibility)
    run_dir = project_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config_used.yaml")
    with open(run_dir / "env.json", "w") as f:
        json.dump({"seed": seed, **env}, f, indent=2)

    print(f"\n✓ Selesai. Checkpoint & log di {run_dir}")
    print(f"  best.pt  : {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
