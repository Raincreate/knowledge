"""Resolve the local model assets independently of the working directory."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "models"
BGE_M3_PATH = MODEL_ROOT / "bge-m3"
BGE_RERANKER_PATH = MODEL_ROOT / "bge-reranker-v2-m3"
HF_HOME = PROJECT_ROOT / ".huggingface-cache"


def _layout_reader_path() -> Path:
    snapshots = HF_HOME / "hub" / "models--hantian--layoutreader" / "snapshots"
    candidates = sorted(path for path in snapshots.glob("*") if path.is_dir())
    return candidates[-1] if candidates else snapshots


LAYOUT_READER_PATH = _layout_reader_path()


def configure_local_model_environment() -> None:
    """Expose absolute paths for all copied model assets."""
    paths = {
        "MODELSCOPE_CACHE": PROJECT_ROOT / ".modelscope-cache",
        "HF_HOME": HF_HOME,
        "BGE_M3_PATH": BGE_M3_PATH,
        "BGE_RERANKER_LARGE": BGE_RERANKER_PATH,
        "LAYOUT_READER_PATH": LAYOUT_READER_PATH,
    }
    for key, default_path in paths.items():
        configured = Path(os.environ.get(key, str(default_path))).expanduser()
        if not configured.is_absolute():
            configured = PROJECT_ROOT / configured
        os.environ[key] = str(configured.resolve())


def validate_local_models() -> dict[str, Path]:
    """Return the complete local models, raising if any model is incomplete."""
    assets = {
        "bge_m3_pytorch": BGE_M3_PATH / "pytorch_model.bin",
        "bge_reranker_v2_m3": BGE_RERANKER_PATH / "model.safetensors",
        "layoutreader": LAYOUT_READER_PATH / "model.safetensors",
    }
    missing = {name: path for name, path in assets.items() if not path.is_file()}
    if missing:
        details = ", ".join(f"{name}={path}" for name, path in missing.items())
        raise FileNotFoundError(f"Local model assets are incomplete: {details}")
    return assets
