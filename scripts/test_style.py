"""Phase 4 smoke test: StyleRefiner re-textures a hole locally, within 6 GB VRAM.

Runs the diffusion img2img path with NO LoRA (base SD 1.5) so it verifies the
plumbing, locality, and VRAM envelope *before* the artist LoRA exists. First run
downloads the base SD 1.5 weights (~4 GB) to ~/.cache/huggingface.
"""

import cv2
import numpy as np
from PIL import Image

from samsplit.config import EXAMPLES_DIR, OUTPUTS_DIR, Config
from samsplit.pipeline import build_clean_plate
from samsplit.style import StyleRefiner

cfg = Config()
arr = np.asarray(Image.open(EXAMPLES_DIR / "synthetic.png").convert("RGB"))
h, w = arr.shape[:2]

# A synthetic dis-occlusion hole (filled circle) in the lower-left quadrant. In real
# use this is the union of committed masks, already LaMa-filled; here we just need a
# region for the refiner to restyle.
hole = np.zeros((h, w), np.uint8)
cv2.circle(hole, (w // 3, 2 * h // 3), max(20, min(h, w) // 6), 1, thickness=-1)
hole_bool = hole.astype(bool)

# None = base style only (no LoRA trained yet) — enough to exercise the path.
refiner = StyleRefiner(
    cfg.style_base_id, None, cfg.device,
    max_side=cfg.style_max_side, prompt=cfg.style_prompt,
    negative_prompt=cfg.style_negative_prompt, guidance=cfg.style_guidance, seed=cfg.style_seed,
)

if cfg.device == "cuda":
    import torch

    torch.cuda.reset_peak_memory_stats()

out = refiner.refine(arr, hole_bool, dilate_px=4, strength=0.4, steps=12)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
Image.fromarray(out).save(OUTPUTS_DIR / "test_style_refined.png")

assert out.shape == arr.shape, "refine changed the canvas shape"
assert out.dtype == np.uint8, "refine should return uint8"

changed = np.any(out != arr, axis=2)
in_hole = float(changed[hole_bool].mean())
overall = float(changed.mean())
print(f"hole pixels changed: {in_hole * 100:.1f}% | whole-canvas changed: {overall * 100:.1f}%")
assert in_hole > 0.8, f"style refine barely touched the hole ({in_hole:.2f})"
assert overall < 0.6, f"style refine leaked far outside the hole ({overall:.2f}) — not local"

if cfg.device == "cuda":
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"peak VRAM: {peak_gb:.2f} GB")
    assert peak_gb < 6.0, f"style refine exceeded 6 GB ({peak_gb:.2f} GB)"

# build_clean_plate must route through the refiner and stay hole-local. A pass-through
# inpainter keeps this fast (no LaMa download) while exercising the real pipeline
# function the UI's preview/export handlers call.
class _PassInpainter:
    def inpaint(self, image_rgb, mask, dilate_px=4):
        return np.ascontiguousarray(image_rgb[:, :, :3]).astype(np.uint8)


plate = build_clean_plate(arr, [hole_bool], _PassInpainter(), dilate_px=4,
                          refiner=refiner, style_strength=0.4, style_steps=12)
pchanged = np.any(plate != arr, axis=2)
print(f"build_clean_plate hole changed: {pchanged[hole_bool].mean() * 100:.1f}% | "
      f"canvas changed: {pchanged.mean() * 100:.1f}%")
assert plate.shape == arr.shape, "build_clean_plate changed the canvas shape"
assert pchanged[hole_bool].mean() > 0.8, "build_clean_plate refiner didn't restyle the hole"
assert pchanged.mean() < 0.6, "build_clean_plate refiner leaked outside the hole"

print("OK — StyleRefiner + build_clean_plate wiring verified; plate -> outputs/test_style_refined.png")
