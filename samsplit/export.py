"""Export layers as full-canvas RGBA PNGs + manifest.json + an After Effects
import script (``import_to_AE.jsx``).

Why full-canvas PNGs: every layer is the same W x H as the source with its
element in the original position on transparency. In AE they stack at the same
position with zero offset -- pixel-perfect reassembly, and any layer can then be
moved/animated freely. With ``parallax=True`` the script instead builds a 2.5D
rig: each layer is a 3D layer at its depth-derived Z (scaled to fill at rest),
plus a camera, so a camera move yields real parallax.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class Layer:
    name: str
    rgba: np.ndarray  # (H, W, 4) uint8, full canvas
    depth_order: int = 0  # 0 = farthest (background); larger = nearer the viewer
    z: float = 0.0  # AE Z position (px); larger = farther from camera
    # element mask (bool) kept for inpainting/depth; never written to disk
    source_mask: np.ndarray | None = field(default=None, repr=False, compare=False)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")


def export_project(
    layers: list[Layer],
    out_dir: str | Path,
    comp_name: str = "SAMSplit",
    fps: float = 24.0,
    duration: float = 10.0,
    parallax: bool = False,
    depth_map: np.ndarray | None = None,
) -> Path:
    """Write PNG layers (back-to-front), manifest.json, import_to_AE.jsx, depth.png."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ordered = sorted(layers, key=lambda layer: layer.depth_order)
    h, w = ordered[0].rgba.shape[:2]

    manifest: dict = {
        "comp_name": comp_name,
        "width": int(w),
        "height": int(h),
        "fps": fps,
        "duration": duration,
        "parallax": bool(parallax),
        "layers": [],
    }
    for i, layer in enumerate(ordered):
        fname = f"{i:02d}_{_safe_name(layer.name) or f'layer_{i:02d}'}.png"
        Image.fromarray(layer.rgba, mode="RGBA").save(out / fname)
        manifest["layers"].append({
            "index": i, "name": layer.name, "file": fname,
            "depth_order": layer.depth_order, "z": round(float(layer.z), 2),
        })

    if depth_map is not None:
        dm = np.clip(np.asarray(depth_map, dtype=np.float32) * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(dm, mode="L").save(out / "depth.png")
        manifest["depth_map"] = "depth.png"

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    jsx = _build_ae_jsx_parallax(manifest) if parallax else _build_ae_jsx(manifest)
    (out / "import_to_AE.jsx").write_text(jsx)
    return out


def _build_ae_jsx(manifest: dict) -> str:
    """Flat stack: import each PNG back-to-front (front-most ends on top)."""
    files_js = ", ".join(json.dumps(layer["file"]) for layer in manifest["layers"])
    return f"""// SAMSplit -> After Effects import (flat stack)
// Run via: File > Scripts > Run Script File...  (select this .jsx, in its folder)
(function () {{
    var COMP_NAME = {json.dumps(manifest["comp_name"])};
    var W = {manifest["width"]}, H = {manifest["height"]};
    var FPS = {manifest["fps"]}, DURATION = {manifest["duration"]};
    var files = [{files_js}];  // back-to-front

    var dir = new File($.fileName).parent;
    app.beginUndoGroup("SAMSplit import");
    var comp = app.project.items.addComp(COMP_NAME, W, H, 1.0, DURATION, FPS);
    comp.openInViewer();

    for (var i = 0; i < files.length; i++) {{
        var f = new File(dir.fsName + "/" + files[i]);
        if (!f.exists) {{ alert("SAMSplit: missing file\\n" + f.fsName); continue; }}
        var item = app.project.importFile(new ImportOptions(f));
        var layer = comp.layers.add(item);          // newest goes on top
        layer.name = files[i].replace(/\\.png$/i, "");
    }}
    app.endUndoGroup();
}})();
"""


def _build_ae_jsx_parallax(manifest: dict) -> str:
    """2.5D rig: 3D layers at depth-derived Z (scale-compensated) + a camera."""
    files_js = ", ".join(json.dumps(layer["file"]) for layer in manifest["layers"])
    zs_js = ", ".join(str(round(float(layer["z"]), 2)) for layer in manifest["layers"])
    return f"""// SAMSplit -> After Effects 2.5D parallax import
// Run via: File > Scripts > Run Script File...  (select this .jsx, in its folder)
// Each layer is a 3D layer placed at its depth-derived Z and scaled to fill at
// rest; animate the "SAMSplit Camera" position to get parallax.
(function () {{
    var COMP_NAME = {json.dumps(manifest["comp_name"])};
    var W = {manifest["width"]}, H = {manifest["height"]};
    var FPS = {manifest["fps"]}, DURATION = {manifest["duration"]};
    var files = [{files_js}];  // back-to-front
    var zs = [{zs_js}];         // matching Z (px); larger = farther

    var dir = new File($.fileName).parent;
    app.beginUndoGroup("SAMSplit 2.5D import");
    var comp = app.project.items.addComp(COMP_NAME, W, H, 1.0, DURATION, FPS);
    comp.openInViewer();

    var cam = comp.layers.addCamera("SAMSplit Camera", [W / 2, H / 2]);
    var D = cam.property("Zoom").value;  // px distance at which 100% scale fills the frame

    for (var i = 0; i < files.length; i++) {{
        var f = new File(dir.fsName + "/" + files[i]);
        if (!f.exists) {{ alert("SAMSplit: missing file\\n" + f.fsName); continue; }}
        var item = app.project.importFile(new ImportOptions(f));
        var layer = comp.layers.add(item);
        layer.threeDLayer = true;
        var z = zs[i];
        layer.property("Position").setValue([W / 2, H / 2, z]);
        var s = (D + z) / D * 100.0;  // scale up so the layer still fills the frame at depth z
        layer.property("Scale").setValue([s, s, s]);
        layer.name = files[i].replace(/\\.png$/i, "");
    }}
    app.endUndoGroup();
}})();
"""
