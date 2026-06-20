"""Util bersama untuk aplikasi Streamlit (demo deteksi + dashboard hasil).

Memuat model terlatih dari experiments/ dan metrik ringkas. Kekhususan:
- YOLOv8s+CBAM butuh registrasi modul CBAM ke parser Ultralytics sebelum di-load.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP = REPO_ROOT / "experiments"
FIGURES = REPO_ROOT / "results" / "figures"
DATA_TEST = REPO_ROOT / "data" / "helmet-roboflow" / "test" / "images"

CLASS_NAMES = ["helmet", "license_plate", "motorcyclist"]

# --- Model untuk DEMO deteksi (checkpoint fine-tuned, seed TERBAIK tiap arsitektur) ---
MODELS = {
    "YOLOv8s (fine-tuned)": {
        "ckpt": "experiments/ft_yolov8s_seed42/weights/best.pt",  # seed42: mAP@0.5=0.9604
        "kind": "yolo",
        "note": "CNN murni — pemenang praktis (terakurat & tercepat).",
    },
    "YOLO11s (fine-tuned)": {
        "ckpt": "experiments/ft_yolo11s_seed0/weights/best.pt",   # seed0: mAP@0.5=0.9589
        "kind": "yolo",
        "note": "CNN dengan atensi parsial (C2PSA).",
    },
    "YOLOv8s+CBAM (fine-tuned)": {
        "ckpt": "experiments/ft_cbam_seed0/weights/best.pt",      # seed0: mAP@0.5=0.9487
        "kind": "yolo",
        "note": "YOLOv8s + modul atensi CBAM (channel+spatial).",
    },
}

# --- Run multi-seed untuk DASHBOARD (rata-rata ± simpangan baku) ---
RUNS = {
    "YOLOv8s": ["ft_yolov8s_seed42", "ft_yolov8s_seed0", "ft_yolov8s_seed1"],
    "YOLO11s": ["ft_yolo11s_seed42", "ft_yolo11s_seed0", "ft_yolo11s_seed1"],
    "YOLOv8s+CBAM": ["ft_cbam_seed42", "ft_cbam_seed0", "ft_cbam_seed1"],
}
# FPS andal dari run tunggal saat GPU senggang (sweep beruntun tidak reliable)
FPS_SINGLE = {"YOLOv8s": 296, "YOLO11s": 189, "YOLOv8s+CBAM": 290}
# mAP@0.5 per-kelas pada baseline YOLOv8s (split uji)
PER_CLASS = {"helmet": 0.923, "license_plate": 0.969, "motorcyclist": 0.990}


def _register_cbam() -> None:
    """Daftarkan CBAM ke namespace parser agar checkpoint CBAM bisa di-deserialize."""
    import ultralytics.nn.tasks as _tasks
    from ultralytics.nn.modules import CBAM as _CBAM

    _tasks.CBAM = _CBAM


def load_model(name: str):
    """Muat model terlatih berdasarkan nama (lihat MODELS). Selalu daftarkan CBAM dulu."""
    spec = MODELS[name]
    ckpt = REPO_ROOT / spec["ckpt"]
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {ckpt}")
    _register_cbam()
    from ultralytics import YOLO

    return YOLO(str(ckpt))


def load_summary() -> list[dict]:
    """Rata-rata ± std metrik per model dari metrics.json multi-seed."""
    rows = []
    for model, runs in RUNS.items():
        m50, m5095 = [], []
        for r in runs:
            p = EXP / r / "metrics.json"
            if p.exists():
                d = json.loads(p.read_text())
                m50.append(d["mAP50"])
                m5095.append(d["mAP50_95"])
        rows.append(
            {
                "Model": model,
                "n_seed": len(m50),
                "mAP@0.5": st.mean(m50) if m50 else float("nan"),
                "mAP@0.5_std": st.stdev(m50) if len(m50) > 1 else 0.0,
                "mAP@[.5:.95]": st.mean(m5095) if m5095 else float("nan"),
                "FPS": FPS_SINGLE.get(model),
            }
        )
    return rows


def sample_images(limit: int = 6) -> list[Path]:
    """Beberapa gambar uji sebagai contoh cepat (kalau dataset tersedia)."""
    if DATA_TEST.exists():
        return sorted(DATA_TEST.glob("*.jpg"))[:limit]
    return sorted((REPO_ROOT / "results" / "predict").glob("*.jpg"))[:limit]
