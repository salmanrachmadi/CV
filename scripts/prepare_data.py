#!/usr/bin/env python
"""Entrypoint: siapkan/validasi dataset deteksi helm (format YOLO).

Contoh:
  # Validasi export manual yang sudah ada di data/helmet-roboflow
  python scripts/prepare_data.py --dataset-dir data/helmet-roboflow

  # Unduh dari Roboflow (butuh ROBOFLOW_API_KEY)
  python scripts/prepare_data.py --roboflow \
      --workspace <ws> --project <proj> --version 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import DATA_ROOT, download_from_roboflow, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Siapkan dataset deteksi helm")
    parser.add_argument(
        "--dataset-dir",
        default=str(DATA_ROOT / "helmet-roboflow"),
        help="Folder dataset (format YOLO Ultralytics)",
    )
    parser.add_argument("--roboflow", action="store_true", help="Unduh via Roboflow SDK")
    parser.add_argument("--workspace", help="Roboflow workspace")
    parser.add_argument("--project", help="Roboflow project")
    parser.add_argument("--version", type=int, help="Roboflow version")
    args = parser.parse_args()

    if args.roboflow:
        if not (args.workspace and args.project and args.version):
            parser.error("--roboflow butuh --workspace, --project, dan --version")
        dataset_dir = download_from_roboflow(
            workspace=args.workspace,
            project=args.project,
            version=args.version,
            dest_name=Path(args.dataset_dir).name,
        )
    else:
        dataset_dir = Path(args.dataset_dir)

    counts = validate_dataset(dataset_dir)
    print(f"\n✓ Dataset siap di {dataset_dir} — total {sum(counts.values())} gambar.")


if __name__ == "__main__":
    main()
