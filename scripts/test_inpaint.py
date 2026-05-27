"""Phase 2 test: remove elements + inpaint the clean plate, verify locality + fill."""

import shutil

import cv2
import numpy as np
from PIL import Image

from samsplit.config import EXAMPLES_DIR, OUTPUTS_DIR, Config
from samsplit.inpaint import LamaInpainter
from samsplit.pipeline import export_project, make_layer, plate_background_layer
from samsplit.segment import Sam2Segmenter

arr = np.asarray(Image.open(EXAMPLES_DIR / "synthetic.png").convert("RGB"))
cfg = Config()

seg = Sam2Segmenter(cfg.sam2_model_id, cfg.device)
seg.set_image(arr)
r_sun = seg.predict(points=[(380, 90)], labels=[1])
r_tree = seg.predict(points=[(160, 280)], labels=[1])

inpainter = LamaInpainter(device=cfg.device)
union = (r_sun.mask | r_tree.mask)
plate = inpainter.inpaint(arr, union, dilate_px=4)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
Image.fromarray(plate).save(OUTPUTS_DIR / "test_inpaint_plate.png")

assert plate.shape == arr.shape, "plate shape mismatch"

# Locality: nothing outside the dilated hole may change.
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))  # matches dilate_px=4
dilated = cv2.dilate(union.astype(np.uint8), k).astype(bool)
assert np.array_equal(plate[~dilated], arr[~dilated]), "inpaint altered pixels outside the hole"
print(f"locality OK — only {dilated.mean() * 100:.1f}% of pixels (the dilated holes) changed")

# Fill happened: the bright-yellow sun region should be repainted (toward sky).
sm = r_sun.mask
orig_sun = arr[sm].astype(int).mean(0)
plate_sun = plate[sm].astype(int).mean(0)
diff_sun = float(np.abs(plate_sun - orig_sun).mean())
print(f"sun region mean RGB {orig_sun.round()} -> {plate_sun.round()} (Δ={diff_sun:.1f})")
assert diff_sun > 20, f"sun region barely changed (Δ={diff_sun}) — inpaint likely failed"
assert plate_sun[2] > orig_sun[2] + 15, "sun fill should trend bluer (toward sky)"

# Export with the inpainted plate as background.
sun_layer = make_layer(arr, r_sun.mask, "sun", 1)
tree_layer = make_layer(arr, r_tree.mask, "tree", 2)
layers = [plate_background_layer(plate), sun_layer, tree_layer]
out_dir = OUTPUTS_DIR / "test_inpaint"
if out_dir.exists():
    shutil.rmtree(out_dir)
export_project(layers, out_dir, comp_name="InpaintTest")
pngs = sorted(p.name for p in out_dir.iterdir() if p.suffix == ".png")
print("exported:", pngs)
assert pngs[0].startswith("00_background_plate"), "plate should be the bottom layer"


def over(dst, rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return rgba[..., :3].astype(np.float32) * a + dst * (1.0 - a)


comp = plate.astype(np.float32)
for layer in (sun_layer, tree_layer):
    comp = over(comp, layer.rgba.astype(np.float32))
mad = float(np.abs(comp - arr).mean())
print(f"recomposition (plate+elements) vs original mean-abs-diff: {mad:.2f} (small = only halo ring differs)")
assert mad < 5.0, "plate+elements should reconstruct the original except a thin halo ring"
print("OK — Phase 2 dis-occlusion verified; plate saved to outputs/test_inpaint_plate.png")
