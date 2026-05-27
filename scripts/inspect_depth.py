"""Probe Depth Anything V2: API shape + near/far convention (downloads ~100MB once)."""

import numpy as np
import torch
from PIL import Image
from transformers import pipeline

from samsplit.config import EXAMPLES_DIR

device = 0 if torch.cuda.is_available() else -1
pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)

img = Image.open(EXAMPLES_DIR / "synthetic.png").convert("RGB")
out = pipe(img)
print("output keys:", list(out.keys()))

pd = out["predicted_depth"]
d = pd.squeeze().detach().cpu().numpy().astype(np.float32)
print("predicted_depth", tuple(pd.shape), "-> squeezed", d.shape, "| image", img.size)
print(f"range: min {d.min():.3f} max {d.max():.3f}")

top = float(d[: d.shape[0] // 5].mean())
bot = float(d[-d.shape[0] // 5:].mean())
print(f"top-fifth mean {top:.3f} | bottom-fifth mean {bot:.3f}")
print("convention:", "larger = NEARER (bottom>top as expected)" if bot > top
      else "larger = FARTHER (top>bottom) -- will need invert")
