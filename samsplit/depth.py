"""Monocular depth via Depth Anything V2 -> a nearness map (1.0 = nearest).

Used to auto-order layers back-to-front and to assign each layer a Z position
for the 2.5D parallax rig. Depth Anything outputs inverse depth (larger = nearer).
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


class DepthEstimator:
    def __init__(self, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str | None = None):
        self.model_id = model_id
        self._device = device
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from transformers import pipeline

            dev = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            idx = 0 if str(dev).startswith("cuda") else -1
            self._pipe = pipeline("depth-estimation", model=self.model_id, device=idx)
        return self._pipe

    def nearness_map(self, image_rgb: np.ndarray) -> np.ndarray:
        """Image (H,W,3) -> float32 (H,W) in [0,1], where 1.0 is nearest."""
        pipe = self._load()
        h, w = image_rgb.shape[:2]
        out = pipe(Image.fromarray(np.ascontiguousarray(image_rgb[:, :, :3])))
        d = out["predicted_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
        if d.shape[:2] != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = float(d.min()), float(d.max())
        if hi - lo < 1e-6:
            return np.zeros((h, w), np.float32)
        return ((d - lo) / (hi - lo)).astype(np.float32)
