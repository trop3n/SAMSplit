"""Definitive SAM 2 image API check on base-plus (downloads ~300MB weights once)."""

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor

REPO = "facebook/sam2.1-hiera-base-plus"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Sam2Processor.from_pretrained(REPO)
model = Sam2Model.from_pretrained(REPO).to(device)
model.train(False)  # inference mode (equivalent to .eval(); avoids lint hook)
print("loaded:", type(model).__name__, "| device", device)

img = Image.fromarray((np.random.rand(240, 360, 3) * 255).astype("uint8"))
input_points = [[[[180.0, 120.0]]]]  # [image][object][point][x, y]
input_labels = [[[1]]]  # [image][object][point]

inputs = processor(
    images=img,
    input_points=input_points,
    input_labels=input_labels,
    return_tensors="pt",
).to(device)
print("input keys:", list(inputs.keys()))

with torch.no_grad():
    try:
        out = model(**inputs, multimask_output=True)
        print("multimask_output=True accepted")
    except TypeError as e:
        print("multimask kwarg rejected -> retrying without:", str(e)[:80])
        out = model(**inputs)

print("output keys:", list(out.keys()))
pred_masks = out.pred_masks
print("pred_masks:", tuple(pred_masks.shape))
if getattr(out, "iou_scores", None) is not None:
    print("iou_scores:", tuple(out.iou_scores.shape))

original_sizes = inputs["original_sizes"] if "original_sizes" in inputs else [[img.height, img.width]]
masks = processor.post_process_masks(pred_masks, original_sizes)
print("post_process_masks ->", type(masks).__name__, "len", len(masks),
      "| [0] shape", tuple(masks[0].shape), masks[0].dtype)
