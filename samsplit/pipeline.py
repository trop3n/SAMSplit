"""Compositing orchestration: turn (image, mask) commits into export layers."""

from __future__ import annotations

import numpy as np

from samsplit.export import Layer, export_project
from samsplit.matte import (
    mask_to_trimap,
    soft_alpha_from_mask,
    to_full_canvas_rgba,
    union_mask,
)

__all__ = [
    "make_layer",
    "full_background_layer",
    "build_clean_plate",
    "plate_background_layer",
    "assign_depth_order_and_z",
    "export_project",
    "Layer",
]


def make_layer(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    name: str,
    depth_order: int,
    *,
    edge_mode: str = "matte",  # "matte" (ViTMatte) | "feather"
    feather: float = 2.0,
    erode_px: int = 1,
    band_px: int = 12,
    matter=None,
) -> Layer:
    """Cut an element onto a full-canvas RGBA layer using the chosen edge mode."""
    if edge_mode == "matte" and matter is not None:
        alpha = matter.alpha(image_rgb, mask_to_trimap(mask, band_px=band_px))
    else:
        alpha = soft_alpha_from_mask(mask, feather=feather, erode_px=erode_px)
    rgba = to_full_canvas_rgba(image_rgb, alpha)
    return Layer(name=name, rgba=rgba, depth_order=depth_order,
                 source_mask=np.asarray(mask, dtype=bool))


def full_background_layer(image_rgb, name="background_full", depth_order=0) -> Layer:
    """The whole source image as an opaque bottom layer (no hole-fill)."""
    h, w = image_rgb.shape[:2]
    alpha = np.full((h, w), 255, dtype=np.uint8)
    rgba = np.dstack([image_rgb[:, :, :3], alpha]).astype(np.uint8)
    return Layer(name=name, rgba=rgba, depth_order=depth_order)


def build_clean_plate(image_rgb, masks, inpainter, dilate_px: int = 4,
                      refiner=None, *, style_strength: float = 0.3, style_steps: int = 15) -> np.ndarray:
    """Remove every element (union of masks) and inpaint -> clean background RGB.

    With ``refiner`` set (Phase 4 style-matched fill), the inpainted hole is then
    re-textured with the artist style LoRA so the invented paint reads as the
    artist's own hand. The refine is confined to the hole, so the rest of the plate
    stays identical to LaMa's output. ``refiner=None`` is exactly the Phase 2 behavior.
    """
    hole = union_mask(list(masks))
    plate = inpainter.inpaint(image_rgb, hole, dilate_px=dilate_px)
    if refiner is not None:
        plate = refiner.refine(plate, hole, dilate_px=dilate_px,
                               strength=style_strength, steps=style_steps)
    return plate


def plate_background_layer(plate_rgb, name="background_plate", depth_order=0) -> Layer:
    """An inpainted clean plate as the opaque bottom layer (dis-occlusion fill)."""
    h, w = plate_rgb.shape[:2]
    alpha = np.full((h, w), 255, dtype=np.uint8)
    rgba = np.dstack([plate_rgb[:, :, :3], alpha]).astype(np.uint8)
    return Layer(name=name, rgba=rgba, depth_order=depth_order)


def assign_depth_order_and_z(element_layers, nearness, strength_px, reverse=False) -> list[float]:
    """Set each element layer's depth_order (back-to-front) and Z from its median
    nearness over its mask. Returns the per-layer nearness used (for debugging).

    nearness is in [0,1] with 1 = nearest. Nearer layers get higher depth_order
    (drawn on top) and smaller Z (closer to camera).
    """
    near: list[float] = []
    for layer in element_layers:
        m = layer.source_mask
        v = float(np.median(nearness[m])) if (m is not None and bool(m.any())) else 0.0
        near.append(1.0 - v if reverse else v)

    for rank, idx in enumerate(np.argsort(near)):  # ascending -> farthest gets the lowest order
        element_layers[int(idx)].depth_order = rank + 1
    for i, layer in enumerate(element_layers):
        layer.z = float((1.0 - near[i]) * strength_px)
    return near
