"""Validate layered PSD writing on this env (pytoshop) + flattened composite."""

import numpy as np
import pytoshop
from PIL import Image

from samsplit.config import OUTPUTS_DIR
from samsplit.export import Layer, export_psd, flatten_layers

H, W = 120, 160
back = np.zeros((H, W, 4), np.uint8)
back[..., 0] = 200
back[..., 3] = 255  # opaque red background
mid = np.zeros((H, W, 4), np.uint8)
mid[40:90, 40:120, 1] = 200
mid[40:90, 40:120, 3] = 255  # green square
front = np.zeros((H, W, 4), np.uint8)
front[50:80, 60:100, 2] = 220
front[50:80, 60:100, 3] = 255  # blue square
layers = [Layer("back", back, 0), Layer("mid", mid, 1), Layer("front", front, 2)]

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
path = OUTPUTS_DIR / "test_layers.psd"
export_psd(layers, path)

with open(path, "rb") as f:
    psd = pytoshop.read(f)
recs = psd.layer_and_mask_info.layer_info.layer_records
names = [r.name for r in recs]
print("PSD layer order (file stores bottom->top):", names)
assert len(recs) == 3, f"expected 3 layers, got {len(recs)}"

im = Image.open(path)
print("PIL merged image:", im.size, im.mode)
assert im.size == (W, H), "merged size mismatch"

comp = flatten_layers(layers)
assert comp.shape == (H, W, 3), "composite shape wrong"
print("composite shape:", comp.shape)
print("OK — layered PSD writes, parses (3 layers), opens in PIL; composite flattens")
