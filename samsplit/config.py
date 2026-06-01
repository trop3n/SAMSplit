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
STYLE_LORA_DIR = MODELS_DIR / "style_lora"  # Phase 4: artist style LoRA weights live here

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

# Phase 4 — style-matched dis-occlusion fill. A LoRA fine-tuned on the artist's own
# catalog is loaded into a base SD 1.5 img2img pipeline to re-impose the artist's
# brushwork on the LaMa fill. Train the LoRA on the BASE model below: its attention
# layers are shared with the inpaint/img2img UNet, so the adapter transfers.
STYLE_BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
# A distinctive trigger baked into the training captions AND the refine prompt, so
# the LoRA's learned style activates at inference. Keep these two in lockstep.
STYLE_TRIGGER = "bas_atmospheric"


def detect_device() -> str:
    """Return 'cuda' if a GPU is usable, else 'cpu'."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def find_style_lora(directory: Path | None = None) -> Path | None:
    """Return the first .safetensors LoRA under STYLE_LORA_DIR, or None if absent.

    Used to gate the style-fill feature: it is only offered once weights exist.
    """
    directory = directory or STYLE_LORA_DIR
    if not directory.exists():
        return None
    weights = sorted(directory.glob("*.safetensors"))
    return weights[0] if weights else None


@dataclass
class Config:
    sam2_size: str = "base"  # small | base | large
    depth_size: str = "small"  # small | base
    device: str = field(default_factory=detect_device)
    # 6 GB VRAM: load one large model at a time and free it before the next.
    low_vram: bool = True

    # --- Phase 4 style-matched fill (inactive until a LoRA is trained) ---
    style_base_id: str = STYLE_BASE_MODEL
    style_strength: float = 0.25  # img2img strength; low keeps LaMa structure, re-textures only
    style_steps: int = 15
    style_guidance: float = 5.0
    style_max_side: int = 768  # cap the refine crop for 6 GB VRAM
    style_seed: int = 0
    style_prompt: str = (
        f"a painting in {STYLE_TRIGGER} style, atmospheric cinematic landscape, "
        "soft dramatic light, painterly"
    )
    style_negative_prompt: str = "photograph, 3d render, sharp focus, text, watermark, frame"

    @property
    def sam2_model_id(self) -> str:
        return SAM2_MODELS[self.sam2_size]

    @property
    def depth_model_id(self) -> str:
        return DEPTH_MODELS[self.depth_size]

    @property
    def style_lora_path(self) -> Path | None:
        """Path to the trained artist LoRA, or None if it hasn't been trained yet."""
        return find_style_lora()
