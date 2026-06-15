"""Penyiapan & validasi dataset (format YOLO dari Roboflow).

Dua jalur:
  1. Unduh otomatis via Roboflow SDK (butuh ROBOFLOW_API_KEY).
  2. Validasi export manual yang sudah diletakkan di data/<nama>/.

Output yang diharapkan (format Ultralytics):
  data/<nama>/
    data.yaml
    train/images, train/labels
    valid/images, valid/labels
    test/images,  test/labels
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


def load_dotenv(path: Path | None = None) -> None:
    """Muat pasangan KEY=VALUE dari .env ke os.environ (loader ringan, tanpa dependency).

    Hanya mengisi variabel yang belum ada di environment (env asli menang).
    """
    env_path = Path(path) if path else REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def download_from_roboflow(
    workspace: str,
    project: str,
    version: int,
    dest_name: str = "helmet-roboflow",
    fmt: str = "yolov8",
) -> Path:
    """Unduh dataset dari Roboflow ke data/<dest_name>.

    API key diambil dari env var ROBOFLOW_API_KEY (jangan hardcode di repo).
    Otomatis memuat .env bila ada.
    """
    load_dotenv()
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY belum di-set. Tambahkan ke file .env:\n"
            "  ROBOFLOW_API_KEY=<kunci-anda>\n"
            "atau export manual: export ROBOFLOW_API_KEY=<kunci-anda>"
        )

    from roboflow import Roboflow

    dest = DATA_ROOT / dest_name
    dest.parent.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    proj = rf.workspace(workspace).project(project)
    proj.version(version).download(fmt, location=str(dest))
    print(f"✓ Dataset diunduh ke {dest}")
    return dest


def validate_dataset(dataset_dir: Path) -> Dict[str, int]:
    """Validasi struktur dataset & hitung jumlah gambar per split.

    Raises:
        FileNotFoundError: bila data.yaml atau split wajib tidak ada.
    """
    dataset_dir = Path(dataset_dir)
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml tidak ditemukan di {dataset_dir}. "
            "Unduh dulu via Roboflow atau letakkan export manual di sini."
        )

    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names", [])
    print(f"data.yaml OK — {len(names)} kelas: {names}")

    counts: Dict[str, int] = {}
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        if img_dir.exists():
            n = sum(1 for p in img_dir.iterdir() if p.suffix.lower() in image_exts)
            counts[split] = n
            print(f"  {split:6s}: {n} gambar")
        else:
            counts[split] = 0
            print(f"  {split:6s}: (tidak ada) — {img_dir}")

    if counts.get("train", 0) == 0:
        raise FileNotFoundError("Split train kosong — dataset tidak valid untuk training.")
    return counts
