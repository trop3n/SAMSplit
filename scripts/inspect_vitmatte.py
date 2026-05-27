"""Confirm ViTMatte API surface + a loadable repo id (config.json only)."""

import inspect
import json

import transformers
from huggingface_hub import hf_hub_download

print("transformers", transformers.__version__)
print("Matte classes:", sorted(n for n in dir(transformers) if "Matte" in n))

model_cls = getattr(transformers, "VitMatteForImageMatting", None)
if model_cls is not None:
    print("model.forward:", inspect.signature(model_cls.forward))

for proc_name in ("VitMatteImageProcessor", "VitMatteImageProcessorFast"):
    proc_cls = getattr(transformers, proc_name, None)
    if proc_cls is not None:
        print(f"{proc_name}.__call__:", inspect.signature(proc_cls.__call__))

print("\nrepo probe (config.json only):")
for repo in (
    "hustvl/vitmatte-small-composition-1k",
    "hustvl/vitmatte-base-composition-1k",
):
    try:
        cfg = hf_hub_download(repo, "config.json")
        print(f"  OK   {repo}  model_type={json.load(open(cfg)).get('model_type')}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {repo}  {type(e).__name__}: {str(e)[:90]}")
