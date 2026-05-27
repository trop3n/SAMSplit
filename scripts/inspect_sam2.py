"""Confirm the SAM 2 API surface + which HF repo id is transformers-loadable.

Only downloads tiny config.json files (no model weights).
"""

import inspect
import json

import transformers
from huggingface_hub import hf_hub_download

print("transformers", transformers.__version__)
print("Sam2Processor.__call__:", inspect.signature(transformers.Sam2Processor.__call__))
for name in dir(transformers.Sam2Processor):
    if "post_process" in name:
        try:
            print(f"  {name}:", inspect.signature(getattr(transformers.Sam2Processor, name)))
        except (TypeError, ValueError):
            print(f"  {name}: <no signature>")

candidates = [
    "facebook/sam2.1-hiera-base-plus",
    "facebook/sam2-hiera-base-plus",
    "facebook/sam2.1-hiera-small",
    "facebook/sam2.1-hiera-large",
]
print("\nrepo probe (config.json only):")
for repo in candidates:
    try:
        cfg_path = hf_hub_download(repo, "config.json")
        model_type = json.load(open(cfg_path)).get("model_type")
        print(f"  OK   {repo}  model_type={model_type}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {repo}  {type(e).__name__}: {str(e)[:90]}")
