"""Gradio UI for SAMSplit: click -> commit -> matte -> inpaint -> depth -> export.

SAM 2, ViTMatte, LaMa, and Depth Anything load once, lazily, on first use (so
importing this module -- e.g. for tests -- is cheap). For a single local user
these globals are fine; multi-user would need per-session instances.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from samsplit.config import OUTPUTS_DIR, Config
from samsplit.depth import DepthEstimator
from samsplit.export import export_project
from samsplit.inpaint import LamaInpainter
from samsplit.matte import ViTMatteMatter
from samsplit.pipeline import (
    assign_depth_order_and_z,
    build_clean_plate,
    full_background_layer,
    make_layer,
    plate_background_layer,
)
from samsplit.segment import Sam2Segmenter

_cfg = Config()
_segmenter: Sam2Segmenter | None = None
_matter: ViTMatteMatter | None = None
_inpainter: LamaInpainter | None = None
_depther: DepthEstimator | None = None


def get_segmenter() -> Sam2Segmenter:
    global _segmenter
    if _segmenter is None:
        _segmenter = Sam2Segmenter(_cfg.sam2_model_id, _cfg.device)
    return _segmenter


def get_matter() -> ViTMatteMatter:
    global _matter
    if _matter is None:
        _matter = ViTMatteMatter(device=_cfg.device)
    return _matter


def get_inpainter() -> LamaInpainter:
    global _inpainter
    if _inpainter is None:
        _inpainter = LamaInpainter(device=_cfg.device)
    return _inpainter


def get_depther() -> DepthEstimator:
    global _depther
    if _depther is None:
        _depther = DepthEstimator(device=_cfg.device)
    return _depther


# ----------------------------------------------------------------------------- helpers
def render_overlay(orig_rgb, mask, points, labels):
    base = orig_rgb[:, :, :3].astype(np.float32)
    if mask is not None:
        tint = np.array([255.0, 90.0, 0.0], dtype=np.float32)
        base = np.where(mask[..., None], base * 0.55 + tint * 0.45, base)
    out = Image.fromarray(base.clip(0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(out)
    r = max(3, min(orig_rgb.shape[:2]) // 110)
    for (x, y), label in zip(points, labels):
        color = (50, 220, 70) if label == 1 else (235, 45, 45)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255))
    return np.asarray(out)


def _composite_on_gray(rgba):
    rgb = rgba[:, :, :3].astype(np.float32)
    a = rgba[:, :, 3:4].astype(np.float32) / 255.0
    return (rgb * a + np.full_like(rgb, 200.0) * (1.0 - a)).clip(0, 255).astype(np.uint8)


def _gallery(layers):
    return [(_composite_on_gray(layer.rgba), layer.name) for layer in layers]


def _committed_masks(layers):
    return [layer.source_mask for layer in layers if layer.source_mask is not None]


def _strength_px(orig, strength):
    return float(strength) * 0.6 * max(orig.shape[:2])


# ----------------------------------------------------------------------------- handlers
def on_upload(image):
    if image is None:
        return None, None, [], [], None, "Upload an image to begin."
    rgb = np.ascontiguousarray(image[:, :, :3])
    get_segmenter().set_image(rgb)
    msg = (f"Loaded {rgb.shape[1]}×{rgb.shape[0]}. Click an element (Add mode), "
           "refine with +/− clicks, then **Commit layer**.")
    return rgb, rgb, [], [], None, msg


def on_click(mode, points, labels, orig, evt: gr.SelectData):
    if orig is None:
        return None, points, labels, None, "Upload an image first."
    x, y = int(evt.index[0]), int(evt.index[1])
    label = 1 if str(mode).startswith("Add") else 0
    points = list(points) + [(x, y)]
    labels = list(labels) + [label]
    res = get_segmenter().predict(points=points, labels=labels)
    overlay = render_overlay(orig, res.mask, points, labels)
    return (overlay, points, labels, res.mask,
            f"{len(points)} click(s) • predicted IoU {res.score:.2f} • Commit when it looks right.")


def on_clear(orig):
    return orig, [], [], None, "Clicks cleared — click to start a new element."


def on_commit(orig, mask, name, edge_mode, feather, band, layers):
    if orig is None or mask is None:
        return (layers, _gallery(layers), orig, [], [], None, name,
                "Nothing to commit — click an element first.")
    order = len(layers) + 1
    mode = "matte" if str(edge_mode).startswith("Matting") else "feather"
    matter = get_matter() if mode == "matte" else None
    layer = make_layer(orig, mask, (name or "").strip() or f"element_{order}", order,
                       edge_mode=mode, feather=float(feather), band_px=int(band), matter=matter)
    layers = list(layers) + [layer]
    return (layers, _gallery(layers), orig, [], [], None, "",
            f"Committed “{layer.name}” ({mode} edge) — {len(layers)} layer(s). Click the next element.")


def on_preview_plate(orig, layers, dilate):
    masks = _committed_masks(layers)
    if orig is None or not masks:
        return None, "Commit at least one element first, then preview the fill."
    plate = build_clean_plate(orig, masks, get_inpainter(), dilate_px=int(dilate))
    return (plate, f"Background plate: {len(masks)} element(s) removed and inpainted. "
                   "If the fill looks wrong, raise the halo slider or commit tighter masks.")


def on_preview_depth(orig, reverse):
    if orig is None:
        return None, "Upload an image first."
    near = get_depther().nearness_map(orig)
    if reverse:
        near = 1.0 - near
    colored = cv2.applyColorMap((near * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return (cv2.cvtColor(colored, cv2.COLOR_BGR2RGB),
            "Depth (brighter = treated as nearer). Toggle “Reverse depth” if near/far look swapped.")


def on_export(orig, layers, do_inpaint, dilate, auto_order, parallax, strength, reverse):
    if orig is None or not layers:
        return None, "Add at least one committed layer before exporting."
    masks = _committed_masks(layers)
    elements = list(layers)
    strength_px = _strength_px(orig, strength)

    nearness = None
    if auto_order or parallax:
        nearness = get_depther().nearness_map(orig)
        assign_depth_order_and_z(elements, nearness, strength_px, reverse=bool(reverse))

    if do_inpaint and masks:
        background = plate_background_layer(
            build_clean_plate(orig, masks, get_inpainter(), dilate_px=int(dilate)))
        bg_note = "inpainted plate"
    else:
        background = full_background_layer(orig)
        bg_note = "full original"
    background.depth_order = 0
    background.z = strength_px  # farthest

    all_layers = [background] + elements
    out_dir = OUTPUTS_DIR / time.strftime("%Y%m%d-%H%M%S")
    export_project(all_layers, out_dir, parallax=bool(parallax), depth_map=nearness)
    zip_path = shutil.make_archive(str(out_dir), "zip", out_dir)
    rig = "2.5D parallax rig" if parallax else "flat stack"
    return (zip_path,
            f"Exported {len(all_layers)} layers ({rig}, background = {bg_note}) → "
            f"**{Path(zip_path).name}**. Unzip, then in AE run *File ▸ Scripts ▸ Run Script File…* "
            "on `import_to_AE.jsx`" + (" and animate the *SAMSplit Camera* for parallax." if parallax else "."))


# ----------------------------------------------------------------------------- app
def build_app() -> gr.Blocks:
    with gr.Blocks(title="SAMSplit") as demo:
        gr.Markdown(
            "# SAMSplit\n"
            "Isolate elements of a painting, fill behind them, order them by depth, and export "
            "full-canvas PNGs + a one-click After Effects (flat or 2.5D parallax) import."
        )
        orig_state = gr.State(None)
        points_state = gr.State([])
        labels_state = gr.State([])
        mask_state = gr.State(None)
        layers_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=3):
                img = gr.Image(label="Canvas — upload, then click an element",
                               type="numpy", sources=["upload"], interactive=True, height=540)
                status = gr.Markdown("Upload an image to begin.")
            with gr.Column(scale=1):
                mode = gr.Radio(["Add (foreground)", "Remove (background)"],
                                value="Add (foreground)", label="Click mode")
                edge_mode = gr.Radio(["Matting (sharp)", "Feather (fast)"],
                                     value="Matting (sharp)", label="Edge mode")
                band = gr.Slider(2, 40, value=12, step=1, label="Detail band px (matting edge width)")
                feather = gr.Slider(0.0, 8.0, value=2.0, step=0.5, label="Feather σ (Feather mode)")
                name_box = gr.Textbox(label="Layer name", placeholder="e.g. sun, tree, figure")
                with gr.Row():
                    commit_btn = gr.Button("✓ Commit layer", variant="primary")
                    clear_btn = gr.Button("↺ Clear clicks")
                gallery = gr.Gallery(label="Committed layers", columns=3, height=180,
                                     object_fit="contain")

                with gr.Accordion("Background fill", open=True):
                    inpaint_chk = gr.Checkbox(value=True, label="Fill behind elements (inpaint)")
                    dilate = gr.Slider(0, 24, value=4, step=1, label="Halo dilate px")
                    preview_btn = gr.Button("👁 Preview plate")

                with gr.Accordion("Depth & 2.5D", open=True):
                    auto_order = gr.Checkbox(value=True, label="Auto-order layers by depth")
                    parallax_chk = gr.Checkbox(value=True, label="Build 2.5D parallax rig (3D + camera)")
                    strength = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Depth strength (Z spread)")
                    reverse = gr.Checkbox(value=False, label="Reverse depth (flip near/far)")
                    preview_depth_btn = gr.Button("👁 Preview depth")

                export_btn = gr.Button("⬇ Export to AE", variant="primary")
                out_file = gr.File(label="AE package (.zip)")

        with gr.Row():
            plate_view = gr.Image(label="Background plate preview", interactive=False, height=320)
            depth_view = gr.Image(label="Depth preview (brighter = nearer)", interactive=False, height=320)

        img.upload(on_upload, inputs=[img],
                   outputs=[img, orig_state, points_state, labels_state, mask_state, status])
        img.select(on_click, inputs=[mode, points_state, labels_state, orig_state],
                   outputs=[img, points_state, labels_state, mask_state, status])
        clear_btn.click(on_clear, inputs=[orig_state],
                        outputs=[img, points_state, labels_state, mask_state, status])
        commit_btn.click(on_commit,
                         inputs=[orig_state, mask_state, name_box, edge_mode, feather, band, layers_state],
                         outputs=[layers_state, gallery, img, points_state, labels_state,
                                  mask_state, name_box, status])
        preview_btn.click(on_preview_plate, inputs=[orig_state, layers_state, dilate],
                          outputs=[plate_view, status])
        preview_depth_btn.click(on_preview_depth, inputs=[orig_state, reverse],
                                outputs=[depth_view, status])
        export_btn.click(on_export,
                         inputs=[orig_state, layers_state, inpaint_chk, dilate,
                                 auto_order, parallax_chk, strength, reverse],
                         outputs=[out_file, status])
    return demo
