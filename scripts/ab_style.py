"""Phase 4 A/B: LaMa-only fill vs LaMa + artist-LoRA style refine, across pieces.

    uv run python scripts/ab_style.py                       # default painterly set
    uv run python scripts/ab_style.py img1 img2 --strength 0.25
    uv run python scripts/ab_style.py --big                 # larger hole (stress LaMa)

Builds a stacked outputs/ab_style.png — one row per image: [orig+hole | LaMa | LaMa+style].
LaMa and the diffusion+LoRA pipeline load once and are reused across all images.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from samsplit.config import OUTPUTS_DIR, ROOT, Config
from samsplit.inpaint import LamaInpainter
from samsplit.pipeline import build_clean_plate
from samsplit.style import StyleRefiner

DEFAULTS = [
    "04 40 Days 1920 x 1080 BAS-gigapixel-high fidelity v2-2x.jpg",
    "WK02 June 7 Psalm 2 BAS high res.jpg",
    "Parables WK1 BAS 4 High.jpg",
]

ap = argparse.ArgumentParser()
ap.add_argument("images", nargs="*")
ap.add_argument("--strength", type=float, default=None)
ap.add_argument("--prompt", default=None)
ap.add_argument("--big", action="store_true", help="use a larger central hole")
args = ap.parse_args()

cfg = Config()
paths = [Path(p) for p in args.images] if args.images else [ROOT / "ArtTraining" / n for n in DEFAULTS]
strength = cfg.style_strength if args.strength is None else args.strength
prompt = cfg.style_prompt if args.prompt is None else args.prompt

inpainter = LamaInpainter(device=cfg.device)
refiner = StyleRefiner(cfg.style_base_id, cfg.style_lora_path, cfg.device,
                       max_side=cfg.style_max_side, prompt=prompt,
                       negative_prompt=cfg.style_negative_prompt,
                       guidance=cfg.style_guidance, seed=cfg.style_seed)

rows, PANEL = [], 1024
for p in paths:
    img = Image.open(p).convert("RGB")
    w, h = img.size
    s = min(1.0, PANEL / max(w, h))
    arr = np.asarray(img.resize((round(w * s), round(h * s)), Image.LANCZOS))
    H, W = arr.shape[:2]
    if args.big:
        x0, y0, x1, y1 = int(W * 0.30), int(H * 0.28), int(W * 0.74), int(H * 0.86)
    else:
        x0, y0, x1, y1 = int(W * 0.06), int(H * 0.55), int(W * 0.40), int(H * 0.93)
    hole = np.zeros((H, W), np.uint8)
    cv2.rectangle(hole, (x0, y0), (x1, y1), 1, thickness=-1)
    hb = hole.astype(bool)

    lama = build_clean_plate(arr, [hb], inpainter, dilate_px=6)
    styled = build_clean_plate(arr, [hb], inpainter, dilate_px=6, refiner=refiner,
                               style_strength=strength, style_steps=cfg.style_steps)
    ref = arr.copy()
    cv2.rectangle(ref, (x0, y0), (x1, y1), (255, 0, 0), 3)
    vb = np.full((H, 8, 3), 255, np.uint8)
    rows.append(np.concatenate([ref, vb, lama, vb, styled], axis=1))
    diff = float(np.abs(styled[hb].astype(int) - lama[hb].astype(int)).mean())
    print(f"{p.name}: {W}x{H} hole={hb.mean() * 100:.0f}% mean|styled-lama|={diff:.1f}")

maxw = max(r.shape[1] for r in rows)
gap = np.full((12, maxw, 3), 220, np.uint8)
stacked = []
for r in rows:
    if r.shape[1] < maxw:
        r = np.concatenate([r, np.full((r.shape[0], maxw - r.shape[1], 3), 255, np.uint8)], axis=1)
    stacked += [r, gap]
combo = np.concatenate(stacked[:-1], axis=0)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
Image.fromarray(combo).save(OUTPUTS_DIR / "ab_style.png")
print(f"saved outputs/ab_style.png  rows: [orig+hole | LaMa | LaMa+style]  strength={strength}  big={args.big}")
