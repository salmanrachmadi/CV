#!/usr/bin/env python
"""Entrypoint inferensi — deteksi helm pada satu gambar / folder / video.

Contoh:
  python scripts/infer.py \
      --checkpoint experiments/<run>/weights/best.pt \
      --input contoh.jpg --conf 0.25

Hasil visualisasi (bounding box helmet/no-helmet) disimpan ke results/predict/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import print_env

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inferensi deteksi helm")
    parser.add_argument("--checkpoint", required=True, help="Path bobot (best.pt)")
    parser.add_argument("--input", required=True, help="Gambar/folder/video input")
    parser.add_argument("--conf", type=float, default=0.25, help="Ambang confidence")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=0)
    args = parser.parse_args()

    print_env()

    from ultralytics import YOLO

    model = YOLO(args.checkpoint)
    model.predict(
        source=args.input,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project=str(REPO_ROOT / "results"),
        name="predict",
        exist_ok=True,
    )
    print(f"\n✓ Hasil tersimpan di {REPO_ROOT / 'results' / 'predict'}")


if __name__ == "__main__":
    main()
