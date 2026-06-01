"""Gradio UI for SAMSplit: click -> commit -> matte -> inpaint -> depth -> export.

SAM 2, ViTMatte, LaMa, and Depth Anything load once, lazily, on first use (so
importing this module -- e.g. for tests -- is cheap). For a single local user
these globals are fine; multi-user would need per-session instances.
"""

from __future__ import annotations

import json
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
from samsplit.style import StyleRefiner

_cfg = Config()
_segmenter: Sam2Segmenter | None = None
_matter: ViTMatteMatter | None = None
_inpainter: LamaInpainter | None = None
_depther: DepthEstimator | None = None
_refiner: StyleRefiner | None = None


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


def get_refiner() -> StyleRefiner:
    global _refiner
    if _refiner is None:
        _refiner = StyleRefiner(
            _cfg.style_base_id, _cfg.style_lora_path, _cfg.device,
            max_side=_cfg.style_max_side, prompt=_cfg.style_prompt,
            negative_prompt=_cfg.style_negative_prompt, guidance=_cfg.style_guidance,
            seed=_cfg.style_seed,
        )
    return _refiner


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


def on_preview_plate(orig, layers, dilate, use_style=False, style_strength=0.3):
    masks = _committed_masks(layers)
    if orig is None or not masks:
        return None, "Commit at least one element first, then preview the fill."
    refiner = get_refiner() if use_style else None
    plate = build_clean_plate(orig, masks, get_inpainter(), dilate_px=int(dilate),
                              refiner=refiner, style_strength=float(style_strength))
    styled = " + style-matched" if use_style else ""
    return (plate, f"Background plate: {len(masks)} element(s) removed and inpainted{styled}. "
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


def on_export(orig, layers, do_inpaint, dilate, auto_order, parallax, strength, reverse,
              want_psd, want_composite, use_style=False, style_strength=0.3):
    if orig is None or not layers:
        return None, "Add at least one committed layer before exporting."
    masks = _committed_masks(layers)
    elements = list(layers)
    # Re-derive order/Z from commit order on every export. The Layer objects are
    # shared with layers_state, and assign_depth_order_and_z() mutates them in
    # place, so without this reset a prior auto-ordered/parallax export would leak
    # its depth_order/z into a later flat one (auto-order "off" would silently keep
    # the old order). Normalizing first makes each export deterministic.
    for i, layer in enumerate(elements):
        layer.depth_order = i + 1
        layer.z = 0.0
    strength_px = _strength_px(orig, strength)

    nearness = None
    if auto_order or parallax:
        nearness = get_depther().nearness_map(orig)
        assign_depth_order_and_z(elements, nearness, strength_px, reverse=bool(reverse))

    if do_inpaint and masks:
        refiner = get_refiner() if use_style else None
        background = plate_background_layer(
            build_clean_plate(orig, masks, get_inpainter(), dilate_px=int(dilate),
                              refiner=refiner, style_strength=float(style_strength)))
        bg_note = "style-matched plate" if use_style else "inpainted plate"
    else:
        background = full_background_layer(orig)
        bg_note = "full original"
    background.depth_order = 0
    background.z = strength_px

    all_layers = [background] + elements
    out_dir = OUTPUTS_DIR / time.strftime("%Y%m%d-%H%M%S")
    export_project(all_layers, out_dir, parallax=bool(parallax), depth_map=nearness,
                   write_psd=bool(want_psd), write_composite=bool(want_composite))
    manifest = json.loads((out_dir / "manifest.json").read_text())
    zip_path = shutil.make_archive(str(out_dir), "zip", out_dir)

    extras = []
    if "composite" in manifest:
        extras.append("composite .png/.jpg")
    if "psd" in manifest:
        extras.append("layers.psd")
    elif "psd_error" in manifest:
        extras.append("⚠ .psd failed (see manifest)")
    extra_note = (" • " + ", ".join(extras)) if extras else ""
    rig = "2.5D parallax rig" if parallax else "flat stack"
    return (zip_path,
            f"Exported {len(all_layers)} layers ({rig}, background = {bg_note}){extra_note} → "
            f"**{Path(zip_path).name}**. Open `layers.psd` in Photoshop, or run `import_to_AE.jsx` in AE"
            + (" then animate the *SAMSplit Camera*." if parallax else "."))


# ----------------------------------------------------------------------------- app
def build_app() -> gr.Blocks:
    with gr.Blocks(title="SAMSplit") as demo:
        gr.Markdown(
            "# SAMSplit\n"
            "Isolate elements of a painting, fill behind them, order them by depth, and export "
            "layered PSD / PNG layers + a one-click After Effects (flat or 2.5D parallax) import."
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
                gallery = gr.Gallery(label="Committed layers", columns=3, height=170,
                                     object_fit="contain")

                with gr.Accordion("Background fill", open=False):
                    inpaint_chk = gr.Checkbox(value=True, label="Fill behind elements (inpaint)")
                    dilate = gr.Slider(0, 24, value=4, step=1, label="Halo dilate px")
                    _lora_ready = _cfg.style_lora_path is not None
                    style_chk = gr.Checkbox(
                        value=False, interactive=_lora_ready,
                        label="Style-match fill (artist LoRA)",
                        info=("Re-paint the fill in the artist's style."
                              if _lora_ready else
                              "Train a LoRA into models/style_lora/ to enable (Phase 4)."))
                    style_strength = gr.Slider(0.0, 0.8, value=_cfg.style_strength, step=0.05,
                                               label="Style strength", visible=_lora_ready)
                    preview_btn = gr.Button("👁 Preview plate")

                with gr.Accordion("Depth & 2.5D", open=False):
                    auto_order = gr.Checkbox(value=True, label="Auto-order layers by depth")
                    parallax_chk = gr.Checkbox(value=True, label="Build 2.5D parallax rig (3D + camera)")
                    strength = gr.Slider(0.0, 1.0, value=0.4, step=0.05, label="Depth strength (Z spread)")
                    reverse = gr.Checkbox(value=False, label="Reverse depth (flip near/far)")
                    preview_depth_btn = gr.Button("👁 Preview depth")

                with gr.Accordion("Also export", open=True):
                    psd_chk = gr.Checkbox(value=True, label="Layered Photoshop .psd")
                    composite_chk = gr.Checkbox(value=True, label="Flattened composite (.png + .jpg)")

                export_btn = gr.Button("⬇ Export", variant="primary")
                out_file = gr.File(label="Export package (.zip)")

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
        preview_btn.click(on_preview_plate,
                          inputs=[orig_state, layers_state, dilate, style_chk, style_strength],
                          outputs=[plate_view, status])
        preview_depth_btn.click(on_preview_depth, inputs=[orig_state, reverse],
                                outputs=[depth_view, status])
        export_btn.click(on_export,
                         inputs=[orig_state, layers_state, inpaint_chk, dilate, auto_order,
                                 parallax_chk, strength, reverse, psd_chk, composite_chk,
                                 style_chk, style_strength],
                         outputs=[out_file, status])
    return demo
