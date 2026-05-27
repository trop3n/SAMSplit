"""Smoke test for Sam2Segmenter on a synthetic scene (also writes examples/synthetic.png)."""

import time

import numpy as np
from PIL import Image, ImageDraw

from samsplit.config import EXAMPLES_DIR, Config
from samsplit.segment import Sam2Segmenter

H, W = 360, 480
ys = np.linspace(0.0, 1.0, H)[:, None]
sky = np.zeros((H, W, 3), dtype=np.uint8)
sky[..., 0] = np.clip(135 + 80 * ys, 0, 255).astype(np.uint8)
sky[..., 1] = np.clip(180 + 40 * ys, 0, 255).astype(np.uint8)
sky[..., 2] = np.clip(235 - 60 * ys, 0, 255).astype(np.uint8)
img = Image.fromarray(sky)
draw = ImageDraw.Draw(img)
draw.ellipse([330, 40, 430, 140], fill=(250, 220, 60))  # "sun"
draw.polygon([(80, 320), (160, 160), (240, 320)], fill=(40, 120, 50))  # "tree"
arr = np.asarray(img)

EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
img.save(EXAMPLES_DIR / "synthetic.png")

cfg = Config()
print("device:", cfg.device, "| model:", cfg.sam2_model_id)
seg = Sam2Segmenter(cfg.sam2_model_id, cfg.device)

t = time.time()
seg.set_image(arr)
print(f"set_image (encode once): {time.time() - t:.2f}s | embedding cache: {seg._embeddings is not None}")

t = time.time()
r_sun = seg.predict(points=[(380, 90)], labels=[1])
print(f"predict sun:  area={int(r_sun.mask.sum()):6d}px score={r_sun.score:.3f} t={time.time() - t:.3f}s")

t = time.time()
r_tree = seg.predict(points=[(160, 280)], labels=[1])
print(f"predict tree: area={int(r_tree.mask.sum()):6d}px score={r_tree.score:.3f} t={time.time() - t:.3f}s")

total = H * W
print(f"image={W}x{H} ({total}px); sun frac={r_sun.mask.mean():.3f} tree frac={r_tree.mask.mean():.3f}")
assert 0 < r_sun.mask.sum() < total * 0.5, "sun mask implausible"
assert 0 < r_tree.mask.sum() < total * 0.5, "tree mask implausible"
print("OK")
