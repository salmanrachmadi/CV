"""Determinisme: patok seed untuk random, numpy, dan torch.

Sesuai CLAUDE.md — jangan menambah sumber keacakan tanpa seed.
"""
from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Set seed global untuk reproducibility.

    Args:
        seed: nilai seed.
        deterministic: jika True, paksa cuDNN deterministik (lebih lambat,
            tapi hasil dapat diulang).

    Returns:
        seed yang dipakai (untuk dicatat di log run).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
