"""Phase 3 test: depth map -> layer ordering + Z, and parallax/flat export."""

import json
import shutil

import numpy as np
from PIL import Image

from samsplit.config import EXAMPLES_DIR, OUTPUTS_DIR, Config
from samsplit.depth import DepthEstimator
from samsplit.pipeline import (
    assign_depth_order_and_z,
    export_project,
    full_background_layer,
    make_layer,
)
from samsplit.segment import Sam2Segmenter

arr = np.asarray(Image.open(EXAMPLES_DIR / "synthetic.png").convert("RGB"))
H, W = arr.shape[:2]
cfg = Config()

seg = Sam2Segmenter(cfg.sam2_model_id, cfg.device)
seg.set_image(arr)
sun = seg.predict(points=[(380, 90)], labels=[1]).mask
tree = seg.predict(points=[(160, 280)], labels=[1]).mask

nearness = DepthEstimator(device=cfg.device).nearness_map(arr)
assert nearness.shape == (H, W) and 0.0 <= nearness.min() and nearness.max() <= 1.0
print(f"nearness range {nearness.min():.2f}..{nearness.max():.2f}")

sun_layer = make_layer(arr, sun, "sun", 1, edge_mode="feather")
tree_layer = make_layer(arr, tree, "tree", 2, edge_mode="feather")
elements = [sun_layer, tree_layer]

strength = 0.4 * 0.6 * max(H, W)
assign_depth_order_and_z(elements, nearness, strength)
sun_near = float(np.median(nearness[sun]))
tree_near = float(np.median(nearness[tree]))
print(f"median nearness: sun={sun_near:.3f} tree={tree_near:.3f}")
print(f"sun  order={sun_layer.depth_order} z={sun_layer.z:.1f}")
print(f"tree order={tree_layer.depth_order} z={tree_layer.z:.1f}")

nearer, farther = (sun_layer, tree_layer) if sun_near > tree_near else (tree_layer, sun_layer)
assert nearer.depth_order > farther.depth_order, "nearer element should draw on top"
assert nearer.z <= farther.z, "nearer element should have smaller Z"
assert {sun_layer.depth_order, tree_layer.depth_order} == {1, 2}

bg = full_background_layer(arr)
bg.z = strength  # background sits farthest
out = OUTPUTS_DIR / "test_depth"
if out.exists():
    shutil.rmtree(out)
export_project([bg] + elements, out, comp_name="DepthTest", parallax=True, depth_map=nearness)

files = sorted(p.name for p in out.iterdir())
print("exported:", files)
assert "depth.png" in files
manifest = json.loads((out / "manifest.json").read_text())
assert manifest["parallax"] is True and all("z" in layer for layer in manifest["layers"])
jsx = (out / "import_to_AE.jsx").read_text()
for needle in ("addCamera", "threeDLayer", "SAMSplit Camera", "var zs ="):
    assert needle in jsx, f"parallax jsx missing {needle!r}"
dp = Image.open(out / "depth.png")
assert dp.size == (W, H) and dp.mode == "L"

out2 = OUTPUTS_DIR / "test_depth_flat"
if out2.exists():
    shutil.rmtree(out2)
export_project([bg] + elements, out2, parallax=False)
assert "addCamera" not in (out2 / "import_to_AE.jsx").read_text(), "flat export must not add a camera"
print("OK — depth ordering + Z + parallax/flat export verified")
