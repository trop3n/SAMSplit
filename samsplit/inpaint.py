"""Dis-occlusion inpainting with LaMa (simple-lama-inpainting).

Fills the paint *behind* removed elements so a moved layer reveals clean
background instead of the original element. The fill is composited only inside
the (dilated) hole, leaving the rest of the painting pixel-identical.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class LamaInpainter:
    """Lazy wrapper around SimpleLama with a VRAM-safe resolution cap."""

    def __init__(self, device: str | None = None, max_side: int = 1600):
        self.max_side = max_side
        self._device = device
        self._lama = None

    def _model(self):
        if self._lama is None:
            from simple_lama_inpainting import SimpleLama

            if self._device is not None:
                import torch

                self._lama = SimpleLama(device=torch.device(self._device))
            else:
                self._lama = SimpleLama()
        return self._lama

    def inpaint(self, image_rgb: np.ndarray, mask: np.ndarray, dilate_px: int = 4) -> np.ndarray:
        """Fill ``mask`` (True = remove/fill) in ``image_rgb``; return full-res RGB.

        The hole is dilated by ``dilate_px`` so leftover color halo at the element
        edge is also replaced. Pixels outside the dilated hole are preserved.
        """
        rgb = np.ascontiguousarray(image_rgb[:, :, :3]).astype(np.uint8)
        hole = (mask > 0).astype(np.uint8)
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            hole = cv2.dilate(hole, k)
        if int(hole.sum()) == 0:
            return rgb  # nothing to fill

        fill = self._run_lama(rgb, hole)
        out = rgb.copy()
        m = hole.astype(bool)
        out[m] = fill[m]
        return out

    def _run_lama(self, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
        h, w = rgb.shape[:2]
        max_side = self.max_side
        for _ in range(3):
            scale = min(1.0, max_side / max(h, w))
            if scale < 1.0:
                w2, h2 = int(round(w * scale)), int(round(h * scale))
                img_s = cv2.resize(rgb, (w2, h2), interpolation=cv2.INTER_AREA)
                msk_s = cv2.resize(hole * 255, (w2, h2), interpolation=cv2.INTER_NEAREST)
            else:
                img_s, msk_s = rgb, (hole * 255).astype(np.uint8)
            try:
                res_pil = self._model()(Image.fromarray(img_s), Image.fromarray(msk_s.astype(np.uint8)))
                res = np.asarray(res_pil.convert("RGB"))
                if res.shape[:2] != (h, w):
                    res = cv2.resize(res, (w, h), interpolation=cv2.INTER_LINEAR)
                return res
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and max_side > 768:
                    self._empty_cache()
                    max_side = max(768, max_side // 2)
                    continue
                raise
        raise RuntimeError("LaMa inpainting ran out of memory even at reduced resolution")

    @staticmethod
    def _empty_cache():
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
