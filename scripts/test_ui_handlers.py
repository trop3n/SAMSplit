"""Integration test: drive the UI handlers directly (upload->commit->preview->export)."""

import os
import zipfile

import numpy as np
from PIL import Image

from samsplit import ui
from samsplit.config import EXAMPLES_DIR

arr = np.asarray(Image.open(EXAMPLES_DIR / "synthetic.png").convert("RGB"))

# upload -> registers image with the segmenter
_, orig, _, _, _, _ = ui.on_upload(arr)
assert orig is not None

seg = ui.get_segmenter()
m_sun = seg.predict(points=[(380, 90)], labels=[1]).mask
m_tree = seg.predict(points=[(160, 280)], labels=[1]).mask

layers = []
layers = ui.on_commit(orig, m_sun, "sun", "Matting (sharp)", 2.0, 12, layers)[0]
layers = ui.on_commit(orig, m_tree, "tree", "Matting (sharp)", 2.0, 12, layers)[0]
assert len(layers) == 2, f"expected 2 committed layers, got {len(layers)}"
assert all(layer.source_mask is not None for layer in layers), "source_mask not stashed"

plate, msg = ui.on_preview_plate(orig, layers, 4)
assert plate is not None and plate.shape == arr.shape, "preview plate bad shape"
print("preview:", msg.split(".")[0])

# inpaint ON + parallax ON (auto-order, strength 0.4, not reversed)
zip_path, msg = ui.on_export(orig, layers, True, 4, True, True, 0.4, False)
assert zip_path and os.path.exists(zip_path), "export zip missing"
zf = zipfile.ZipFile(zip_path)
names = zf.namelist()
print("zip contents:", names)
assert any("background_plate" in n for n in names), "no inpainted plate in export"
assert "depth.png" in names, "no depth map in parallax export"
assert sum(n.endswith(".png") for n in names) == 4, "expected 3 layers + depth.png"
jsx = zf.read("import_to_AE.jsx").decode()
assert "addCamera" in jsx and "threeDLayer" in jsx, "parallax jsx missing camera/3D"

# depth preview handler
dimg, _ = ui.on_preview_depth(orig, False)
assert dimg is not None and dimg.shape[:2] == orig.shape[:2], "depth preview bad shape"

# inpaint OFF + parallax OFF -> full background, flat jsx (no camera)
zip_path2, _ = ui.on_export(orig, layers, False, 4, False, False, 0.4, False)
zf2 = zipfile.ZipFile(zip_path2)
assert any("background_full" in n for n in zf2.namelist()), "inpaint-off should export full background"
assert "addCamera" not in zf2.read("import_to_AE.jsx").decode(), "flat export must not add a camera"
print("OK — UI handlers verified (plate/full, parallax/flat, depth map + preview)")
