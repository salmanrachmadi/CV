"""Ekstraksi & penyimpanan metrik evaluasi ke CSV/JSON.

Satu sumber kebenaran untuk metrik (CLAUDE.md). Menyimpan mAP@0.5,
mAP@[.5:.95], precision, recall, dan FPS ke experiments/<run>/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def metrics_from_results(results: Any, fps: float | None = None) -> Dict[str, float]:
    """Ambil metrik ringkas dari objek hasil Ultralytics (.val()).

    Kompatibel dengan ultralytics DetMetrics (results.box.*).
    """
    box = results.box
    out: Dict[str, float] = {
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
        "precision": float(box.mp),     # mean precision
        "recall": float(box.mr),        # mean recall
    }
    if fps is not None:
        out["fps"] = float(fps)
    return out


def save_metrics(metrics: Dict[str, float], run_dir: Path) -> None:
    """Tulis metrik ke metrics.json dan metrics.csv di folder run."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame([metrics]).to_csv(run_dir / "metrics.csv", index=False)
    print(f"✓ Metrik disimpan ke {run_dir}/metrics.{{json,csv}}")
    for k, v in metrics.items():
        print(f"    {k:10s}: {v:.4f}")
