"""Quick environment probe: GPU visibility + which SAM backend is available."""

import torch
import transformers

print("torch", torch.__version__, "| cuda_available =", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  device: {torch.cuda.get_device_name(0)} | cuda {torch.version.cuda} "
          f"| vram = {p.total_memory / 1e9:.1f} GB")

print("transformers", transformers.__version__)
print("SAM classes in transformers:", sorted(n for n in dir(transformers) if "Sam" in n))

import gradio
print("gradio", gradio.__version__)

try:
    from simple_lama_inpainting import SimpleLama  # noqa: F401
    print("simple_lama import OK")
except Exception as e:  # pragma: no cover
    print("simple_lama import FAILED:", repr(e))
