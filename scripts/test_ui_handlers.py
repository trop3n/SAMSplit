"""Integration test: drive the UI handlers (upload -> commit -> preview -> export)."""

import io
import os
import zipfile

import numpy as np
import pytoshop
from PIL import Image

from samsplit import ui
from samsplit.config import EXAMPLES_DIR

arr = np.asarray(Image.open(EXAMPLES_DIR / "synthetic.png").convert("RGB"))

_, orig, _, _, _, _ = ui.on_upload(arr)
assert orig is not None

seg = ui.get_segmenter()
m_sun = seg.predict(points=[(380, 90)], labels=[1]).mask
m_tree = seg.predict(points=[(160, 280)], labels=[1]).mask

layers = []
layers = ui.on_commit(orig, m_sun, "sun", "Matting (sharp)", 2.0, 12, layers)[0]
layers = ui.on_commit(orig, m_tree, "tree", "Matting (sharp)", 2.0, 12, layers)[0]
assert len(layers) == 2 and all(layer.source_mask is not None for layer in layers)

plate, msg = ui.on_preview_plate(orig, layers, 4)
assert plate is not None and plate.shape == arr.shape
print("preview:", msg.split(".")[0])

# inpaint ON + parallax ON + PSD + composite
zip_path, _ = ui.on_export(orig, layers, True, 4, True, True, 0.4, False, True, True)
assert zip_path and os.path.exists(zip_path)
zf = zipfile.ZipFile(zip_path)
names = zf.namelist()
print("zip contents:", names)
assert any("background_plate" in n for n in names), "no inpainted plate"
assert "depth.png" in names, "no depth map in parallax export"
assert "layers.psd" in names, "no layered PSD"
assert "composite.png" in names and "composite.jpg" in names, "no flattened composite"
jsx = zf.read("import_to_AE.jsx").decode()
assert "addCamera" in jsx and "threeDLayer" in jsx, "parallax jsx missing camera/3D"

# PSD parses with background + 2 elements = 3 layers
psd = pytoshop.read(io.BytesIO(zf.read("layers.psd")))
nlayers = len(psd.layer_and_mask_info.layer_info.layer_records)
print("psd layers:", nlayers)
assert nlayers == 3, f"expected 3 PSD layers, got {nlayers}"

dimg, _ = ui.on_preview_depth(orig, False)
assert dimg is not None and dimg.shape[:2] == orig.shape[:2]

# inpaint OFF + parallax OFF -> full background, flat jsx (no camera), still psd+composite
zip_path2, _ = ui.on_export(orig, layers, False, 4, False, False, 0.4, False, True, True)
zf2 = zipfile.ZipFile(zip_path2)
names2 = zf2.namelist()
assert any("background_full" in n for n in names2), "inpaint-off should export full background"
assert "addCamera" not in zf2.read("import_to_AE.jsx").decode(), "flat export must not add a camera"
assert "layers.psd" in names2 and "composite.png" in names2, "psd/composite missing in flat export"
print("OK — handlers verified (plate/full, parallax/flat, depth, PSD + composite)")
