"""End-to-end Phase 1 test: segment -> layers -> export, then verify the package."""

import json
import shutil

import numpy as np
from PIL import Image

from samsplit.config import EXAMPLES_DIR, OUTPUTS_DIR, Config
from samsplit.pipeline import export_project, full_background_layer, make_layer
from samsplit.segment import Sam2Segmenter

arr = np.asarray(Image.open(EXAMPLES_DIR / "synthetic.png").convert("RGB"))
H, W = arr.shape[:2]

cfg = Config()
seg = Sam2Segmenter(cfg.sam2_model_id, cfg.device)
seg.set_image(arr)
r_sun = seg.predict(points=[(380, 90)], labels=[1])
r_tree = seg.predict(points=[(160, 280)], labels=[1])

layers = [
    full_background_layer(arr),
    make_layer(arr, r_sun.mask, "sun", 1, feather=2.0),
    make_layer(arr, r_tree.mask, "tree", 2, feather=2.0),
]
out_dir = OUTPUTS_DIR / "test_export"
if out_dir.exists():
    shutil.rmtree(out_dir)
export_project(layers, out_dir, comp_name="SynthTest")

print("exported:", sorted(p.name for p in out_dir.iterdir()))
pngs = sorted(p for p in out_dir.iterdir() if p.suffix == ".png")
assert len(pngs) == 3, f"expected 3 PNGs, got {len(pngs)}"
for p in pngs:
    im = Image.open(p)
    assert im.mode == "RGBA", f"{p.name} not RGBA"
    assert im.size == (W, H), f"{p.name} size {im.size} != {(W, H)}"
print(f"all layers full-canvas RGBA {(W, H)}")

manifest = json.loads((out_dir / "manifest.json").read_text())
assert manifest["width"] == W and manifest["height"] == H
assert [layer["name"] for layer in manifest["layers"]] == ["background_full", "sun", "tree"]
print("manifest OK:", manifest["comp_name"], [layer["file"] for layer in manifest["layers"]])

jsx = (out_dir / "import_to_AE.jsx").read_text()
for needle in ["addComp", "SynthTest", "01_sun.png", "02_tree.png", "importFile"]:
    assert needle in jsx, f"jsx missing {needle!r}"
print(f"import_to_AE.jsx OK ({len(jsx)} chars)")


def alpha_frac(name):
    return float((np.asarray(Image.open(out_dir / name))[..., 3] > 0).mean())


bg_a = alpha_frac(manifest["layers"][0]["file"])
sun_a = alpha_frac("01_sun.png")
assert bg_a > 0.999, f"background should be opaque, got {bg_a}"
assert 0.0 < sun_a < 0.5, f"sun coverage implausible: {sun_a}"
print(f"alpha coverage: bg={bg_a:.3f} sun={sun_a:.3f} tree={alpha_frac('02_tree.png'):.3f}")


def over(dst_rgb, rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return rgba[..., :3].astype(np.float32) * a + dst_rgb * (1.0 - a)


comp = arr.astype(np.float32)  # background layer == full original
for name in ["01_sun.png", "02_tree.png"]:
    comp = over(comp, np.asarray(Image.open(out_dir / name)).astype(np.float32))
mad = float(np.abs(comp - arr).mean())
print(f"recomposition mean-abs-diff vs original: {mad:.3f} (expect ~0)")
assert mad < 1.0, "layers do not recompose to the original — offset/alpha bug"
print("OK — Phase 1 end-to-end verified")
