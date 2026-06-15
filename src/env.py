"""Inspeksi lingkungan: device/CUDA dan versi kunci.

Dicatat di tiap run agar eksperimen reproducible (CLAUDE.md: catat versi
python, framework DL, CUDA, dan driver GPU).
"""
from __future__ import annotations

import platform
import sys
from typing import Any, Dict


def collect_env() -> Dict[str, Any]:
    """Kumpulkan info lingkungan sebagai dict yang bisa diserialisasi ke JSON."""
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "torch": None,
        "cuda": None,
    }

    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["device"] = "cuda:0"
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None

    return info


def print_env() -> Dict[str, Any]:
    """Cetak info lingkungan ke stdout dan kembalikan dict-nya."""
    info = collect_env()
    print("=" * 48)
    print("ENVIRONMENT")
    for key, value in info.items():
        print(f"  {key:16s}: {value}")
    print("=" * 48)
    if not info["cuda_available"]:
        print("⚠️  CUDA tidak terdeteksi — training akan berjalan di CPU (lambat).")
    return info


if __name__ == "__main__":
    print_env()
