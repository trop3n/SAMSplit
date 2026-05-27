"""Interactive SAM 2 segmentation.

Wraps transformers' Sam2Model / Sam2Processor behind a small, swappable
interface: ``set_image()`` once, then ``predict(points, labels, box)`` per click.
The image is encoded a single time and its embedding reused, so refinement clicks
are fast (the vision encoder is the expensive part).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import Sam2Model, Sam2Processor


@dataclass
class MaskResult:
    mask: np.ndarray  # (H, W) bool at original resolution
    score: float  # predicted IoU of the chosen mask


def _opt(enc, key):
    """Optional key from a processor BatchEncoding (None if absent)."""
    return enc[key] if key in enc else None


class Sam2Segmenter:
    def __init__(self, model_id: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Sam2Processor.from_pretrained(model_id)
        self.model = Sam2Model.from_pretrained(model_id).to(self.device)
        self.model.train(False)  # inference mode: disables dropout, freezes norms
        self._image: np.ndarray | None = None
        self._embeddings = None

    @torch.no_grad()
    def set_image(self, image_rgb: np.ndarray) -> None:
        """Register the working image and encode it once (reused per click)."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] < 3:
            raise ValueError("set_image expects an (H, W, 3) RGB array")
        self._image = np.ascontiguousarray(image_rgb[:, :, :3])
        enc = self.processor(images=self._image, return_tensors="pt").to(self.device)
        try:
            self._embeddings = self.model.get_image_embeddings(enc["pixel_values"])
        except Exception:  # noqa: BLE001 -- fall back to per-call encoding
            self._embeddings = None

    @torch.no_grad()
    def predict(
        self,
        points: list[tuple[float, float]] | None = None,
        labels: list[int] | None = None,
        box: tuple[float, float, float, float] | None = None,
        multimask: bool = True,
    ) -> MaskResult:
        """Predict a mask from foreground/background points and/or a box.

        ``points`` are (x, y) in original-image pixels; ``labels`` are 1
        (foreground) or 0 (background), one per point. Returns the best of SAM's
        candidate masks by predicted IoU.
        """
        if self._image is None:
            raise RuntimeError("call set_image() before predict()")
        if not points and box is None:
            raise ValueError("predict needs at least one point or a box")
        if points and (labels is None or len(labels) != len(points)):
            raise ValueError("labels must be one per point")

        input_points = [[[[float(x), float(y)] for x, y in points]]] if points else None
        input_labels = [[[int(v) for v in labels]]] if points else None
        input_boxes = [[[float(c) for c in box]]] if box is not None else None

        enc = self.processor(
            images=self._image,
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(self.device)

        out = self._forward(enc, multimask)

        masks = self.processor.post_process_masks(out.pred_masks, enc["original_sizes"])[0]
        # masks: (num_objects, num_masks, H, W) bool -- a single object here.
        scores = out.iou_scores[0, 0]  # (num_masks,)
        best = int(torch.argmax(scores).item()) if multimask else 0
        mask = masks[0, best].detach().cpu().numpy().astype(bool)
        return MaskResult(mask=mask, score=float(scores[best].item()))

    def _forward(self, enc, multimask: bool):
        prompt = dict(
            input_points=_opt(enc, "input_points"),
            input_labels=_opt(enc, "input_labels"),
            input_boxes=_opt(enc, "input_boxes"),
            multimask_output=multimask,
        )
        if self._embeddings is not None:
            try:
                return self.model(image_embeddings=self._embeddings, **prompt)
            except (TypeError, RuntimeError):
                self._embeddings = None  # disable fast path for the rest of the session
        return self.model(pixel_values=enc["pixel_values"], **prompt)
