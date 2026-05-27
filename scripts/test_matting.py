"""Pipeline matting test: ViTMatte layer placement, coverage, edge-adaptiveness."""

import numpy as np
from PIL import Image

from samsplit.config import EXAMPLES_DIR, Config
from samsplit.matte import ViTMatteMatter
from samsplit.pipeline import make_layer
from samsplit.segment import Sam2Segmenter

arr = np.asarray(Image.open(EXAMPLES_DIR / "synthetic.png").convert("RGB"))
cfg = Config()
seg = Sam2Segmenter(cfg.sam2_model_id, cfg.device)
seg.set_image(arr)
sun = seg.predict(points=[(380, 90)], labels=[1]).mask
sun_area = int(sun.sum())

matter = ViTMatteMatter(device=cfg.device)
L_m = make_layer(arr, sun, "sun", 1, edge_mode="matte", band_px=12, matter=matter)
L_f = make_layer(arr, sun, "sun", 1, edge_mode="feather", feather=2.0)

for layer in (L_m, L_f):
    assert layer.rgba.shape == (arr.shape[0], arr.shape[1], 4), "layer not full canvas RGBA"

am = L_m.rgba[..., 3].astype(np.float32) / 255.0
af = L_f.rgba[..., 3].astype(np.float32) / 255.0

cov_m = int((am > 0.5).sum())
print(f"sun mask area={sun_area} | matte cov={cov_m} | feather cov={int((af > 0.5).sum())}")
assert 0.8 * sun_area < cov_m < 1.25 * sun_area, "matte coverage far from the element"

# Placement: matte alpha must be ~0 well outside the element's bbox.
ys, xs = np.where(sun)
outside = np.ones(am.shape, bool)
outside[max(0, ys.min() - 64): ys.max() + 64, max(0, xs.min() - 64): xs.max() + 64] = False
assert am[outside].max() < 0.05, "matte alpha leaked far outside the element"

g_m = float(((am > 0.05) & (am < 0.95)).mean())
g_f = float(((af > 0.05) & (af < 0.95)).mean())
print(f"graded-edge fraction: matte={g_m:.4f} feather={g_f:.4f}")
assert g_m < g_f, "on a hard edge, matting should be crisper than uniform feather"
print("OK — matting layer is correctly placed and edge-adaptive")
