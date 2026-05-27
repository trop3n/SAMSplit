"""Definitive ViTMatte run on a SOFT edge: trimap -> graded alpha that tracks truth."""

import numpy as np
import torch
from PIL import Image
from transformers import VitMatteForImageMatting, VitMatteImageProcessor

REPO = "hustvl/vitmatte-small-composition-1k"
device = "cuda" if torch.cuda.is_available() else "cpu"

proc = VitMatteImageProcessor.from_pretrained(REPO)
model = VitMatteForImageMatting.from_pretrained(REPO).to(device)
model.train(False)

# Disk with a SOFT edge: true alpha ramps 1->0 across a 15px ring (d in 50..65).
H, W = 256, 256
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt((xx - 128.0) ** 2 + (yy - 128.0) ** 2)
true_a = np.clip((65.0 - d) / 15.0, 0.0, 1.0)
fg = np.array([220, 90, 60], float)
bg = np.array([120, 160, 210], float)
img = (fg * true_a[..., None] + bg * (1 - true_a[..., None])).clip(0, 255).astype(np.uint8)

trimap = np.full((H, W), 128, np.uint8)
trimap[d < 46] = 255  # sure foreground
trimap[d > 69] = 0  # sure background

inputs = proc(images=Image.fromarray(img), trimaps=Image.fromarray(trimap), return_tensors="pt").to(device)
print("pixel_values", tuple(inputs["pixel_values"].shape))
with torch.no_grad():
    out = model(**inputs)
a = out.alphas[0, 0].detach().cpu().numpy()[:H, :W]

graded = float(((a > 0.05) & (a < 0.95)).mean())
mae = float(np.abs(a - true_a).mean())
print(f"alpha min {a.min():.3f} max {a.max():.3f} | graded-pixel fraction {graded:.3f} | MAE vs truth {mae:.3f}")
assert a.max() > 0.9 and a.min() < 0.1, "alpha should span ~0..1"
assert graded > 0.02, "expected a soft graded edge band"
assert mae < 0.15, "alpha should track the true soft edge"
print("OK — ViTMatte recovers the soft edge (graded alpha matching ground truth)")
