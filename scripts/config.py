"""Shared paths and device selection for the dl_xview_yolo scripts.

Paths default to the repository checkout so the scripts run anywhere. Override
the root with the DL_XVIEW_ROOT environment variable when the datasets and runs
live outside the repository:

    DL_XVIEW_ROOT=/mnt/data/dl_xview python scripts/train_yolo.py
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("DL_XVIEW_ROOT", Path(__file__).resolve().parents[1]))

DATA = ROOT / "data"
YOLO_DATA = ROOT / "yolo_data"
RUNS = ROOT / "runs"
UI_DIR = ROOT / "ui"
TEST_DIR = DATA / "test_images"
DATA_YAML = YOLO_DATA / "data.yaml"


def select_device() -> str:
    """Return an Ultralytics device string, preferring CUDA, then MPS, then CPU.

    torch is imported lazily so the dataset conversion script stays usable
    without a torch install.
    """
    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    """Human-readable name for a device string returned by select_device()."""
    if device.isdigit():
        import torch

        return torch.cuda.get_device_name(int(device))
    return device.upper()
