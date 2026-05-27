"""Central configuration: model choices, device selection, and paths.

torch is imported lazily inside detect_device() so this module is importable
before the (heavy) dependencies are installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
EXAMPLES_DIR = ROOT / "examples"

# Hugging Face model IDs (auto-downloaded + cached under ~/.cache/huggingface on
# first use). The exact SAM 2 repo ids are verified against the installed
# transformers version at first load — adjust here if a repo id has moved.
SAM2_MODELS = {
    "small": "facebook/sam2.1-hiera-small",
    "base": "facebook/sam2.1-hiera-base-plus",
    "large": "facebook/sam2.1-hiera-large",
}
DEPTH_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
}


def detect_device() -> str:
    """Return 'cuda' if a GPU is usable, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@dataclass
class Config:
    sam2_size: str = "base"  # small | base | large
    depth_size: str = "small"  # small | base
    device: str = field(default_factory=detect_device)
    # 6 GB VRAM: load one large model at a time and free it before the next.
    low_vram: bool = True

    @property
    def sam2_model_id(self) -> str:
        return SAM2_MODELS[self.sam2_size]

    @property
    def depth_model_id(self) -> str:
        return DEPTH_MODELS[self.depth_size]
