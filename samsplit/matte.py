"""Turn a SAM mask into a soft alpha, and compose elements as full-canvas RGBA.

Two edge strategies:
- Gaussian feather (fast, dependency-light) -- good for crisp elements.
- ViTMatte alpha matting -- builds a trimap from the SAM mask and estimates true
  fractional transparency along the edge, recovering soft painterly boundaries.
  It runs on a bbox crop around the element, so cost scales with the element, not
  the whole canvas.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


# ----------------------------------------------------------------------------- feather path
def soft_alpha_from_mask(mask: np.ndarray, feather: float = 2.0, erode_px: int = 1) -> np.ndarray:
    """Binary mask (H, W) -> float alpha (H, W) in [0, 1] with feathered edges."""
    m = (mask > 0).astype(np.uint8) * 255
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1, 2 * erode_px + 1))
        m = cv2.erode(m, k)
    alpha = m.astype(np.float32) / 255.0
    if feather > 0:
        ksize = int(max(3, round(feather * 6)) | 1)
        alpha = cv2.GaussianBlur(alpha, (ksize, ksize), feather)
    return np.clip(alpha, 0.0, 1.0)


# ----------------------------------------------------------------------------- matting path
def mask_to_trimap(mask: np.ndarray, band_px: int = 12) -> np.ndarray:
    """SAM mask -> trimap {0 bg, 128 unknown, 255 fg}.

    The unknown band straddles the SAM edge by ~band_px on each side, giving the
    matting model room to recover the true soft boundary.
    """
    m = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    sure_fg = cv2.erode(m, k)
    sure_bg = cv2.dilate(m, k)
    trimap = np.full(m.shape, 128, np.uint8)
    trimap[sure_bg == 0] = 0
    trimap[sure_fg == 1] = 255
    return trimap


class ViTMatteMatter:
    """Lazy ViTMatte wrapper; estimates soft alpha from an image + trimap."""

    def __init__(self, model_id: str = "hustvl/vitmatte-small-composition-1k",
                 device: str | None = None, max_side: int = 1600):
        self.model_id = model_id
        self.max_side = max_side
        self._device = device
        self._proc = None
        self._model = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._proc = VitMatteImageProcessor.from_pretrained(self.model_id)
            self._model = VitMatteForImageMatting.from_pretrained(self.model_id).to(self._device)
            self._model.train(False)
        return self._proc, self._model

    def alpha(self, image_rgb: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Image (H,W,3) + trimap {0,128,255} -> soft alpha (H,W) float in [0,1]."""
        h, w = trimap.shape[:2]
        rgb = np.ascontiguousarray(image_rgb[:, :, :3]).astype(np.uint8)
        tri = trimap.astype(np.uint8)

        region = tri > 0  # foreground + unknown -> the only area matting must solve
        full = np.zeros((h, w), np.float32)
        if not region.any():
            return full

        ys, xs = np.where(region)
        margin = 24  # include some sure-background context around the band
        y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + 1 + margin)
        x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + 1 + margin)
        crop_alpha = self._alpha_crop(rgb[y0:y1, x0:x1], tri[y0:y1, x0:x1])
        full[y0:y1, x0:x1] = crop_alpha
        return full

    def _alpha_crop(self, rgb: np.ndarray, tri: np.ndarray) -> np.ndarray:
        import torch

        proc, model = self._load()
        ch, cw = tri.shape[:2]
        max_side = self.max_side
        for _ in range(3):
            scale = min(1.0, max_side / max(ch, cw))
            if scale < 1.0:
                cw2, ch2 = int(round(cw * scale)), int(round(ch * scale))
                img_s = cv2.resize(rgb, (cw2, ch2), interpolation=cv2.INTER_AREA)
                tri_s = cv2.resize(tri, (cw2, ch2), interpolation=cv2.INTER_NEAREST)
            else:
                img_s, tri_s = rgb, tri
            try:
                inputs = proc(images=Image.fromarray(img_s),
                              trimaps=Image.fromarray(tri_s), return_tensors="pt").to(self._device)
                with torch.inference_mode():
                    out = model(**inputs)
                a = out.alphas[0, 0].detach().cpu().numpy()[: img_s.shape[0], : img_s.shape[1]]
                if a.shape[:2] != (ch, cw):
                    a = cv2.resize(a, (cw, ch), interpolation=cv2.INTER_LINEAR)
                return np.clip(a, 0.0, 1.0)
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and max_side > 512:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                    max_side = max(512, max_side // 2)
                    continue
                raise
        raise RuntimeError("ViTMatte ran out of memory even at reduced resolution")


# ----------------------------------------------------------------------------- composition
def to_full_canvas_rgba(image_rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compose (H, W, 3) RGB + (H, W) alpha into full-canvas (H, W, 4) RGBA."""
    h, w = alpha.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = image_rgb[..., :3]
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return rgba


def union_mask(masks: list[np.ndarray]) -> np.ndarray:
    """Boolean union of several masks (used to cut every element from the plate)."""
    if not masks:
        raise ValueError("union_mask: no masks given")
    acc = np.zeros_like(masks[0], dtype=bool)
    for m in masks:
        acc |= m > 0
    return acc
