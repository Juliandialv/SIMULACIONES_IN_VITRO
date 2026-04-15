"""Here we are going to define paths and global constants."""

from pathlib import Path

# Project root (this file sits at src/siv/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MESH_DIR = RAW_DIR / "modelsSynth"
