"""Phase 4 step B: train the artist style LoRA locally on the 3050 (6 GB).

Wraps the official diffusers `train_text_to_image_lora.py` (fetched once, matched to
the installed diffusers version so the saved LoRA loads cleanly via
`load_lora_weights`) with a memory budget that fits a 6 GB card.

Prereqs:
    uv sync --extra train          # installs bitsandbytes (8-bit Adam) + datasets
    uv run python scripts/prepare_style_dataset.py /path/to/artist_images
    # close other GPU apps — the SAMSplit app must NOT be running during training

Run:
    uv run python scripts/train_style_lora.py                 # 512 px, rank 16, 1500 steps
    uv run python scripts/train_style_lora.py --resolution 448  # fallback levers if it OOMs
    uv run python scripts/train_style_lora.py --resolution 384 --rank 8

Output: models/style_lora/pytorch_lora_weights.safetensors  (auto-detected by the app).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import urllib.request
from pathlib import Path

import diffusers

from samsplit.config import ROOT, STYLE_BASE_MODEL, STYLE_LORA_DIR

VENDOR = ROOT / "scripts" / "vendor"
TRAINER = VENDOR / "train_text_to_image_lora.py"


def ensure_trainer() -> Path:
    """Fetch the diffusers LoRA trainer matched to the installed version (once)."""
    if TRAINER.exists():
        return TRAINER
    VENDOR.mkdir(parents=True, exist_ok=True)
    ver = f"v{diffusers.__version__}"
    url = (f"https://raw.githubusercontent.com/huggingface/diffusers/{ver}/"
           "examples/text_to_image/train_text_to_image_lora.py")
    print(f"fetching version-matched trainer:\n  {url}")
    try:
        urllib.request.urlretrieve(url, TRAINER)
    except Exception as err:  # noqa: BLE001
        raise SystemExit(
            f"could not download the trainer ({err}).\n"
            f"Save it manually to {TRAINER} from the diffusers {ver} examples "
            "(examples/text_to_image/train_text_to_image_lora.py).")
    return TRAINER


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the artist style LoRA (6 GB budget).")
    ap.add_argument("--data", default=str(ROOT / "data" / "style_dataset"))
    ap.add_argument("--resolution", type=int, default=512, help="drop to 448/384 if OOM")
    ap.add_argument("--rank", type=int, default=16, help="drop to 8 if OOM")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", default="1e-4")
    ap.add_argument("--output", default=str(STYLE_LORA_DIR), help="where to write the LoRA")
    ap.add_argument("--checkpointing-steps", type=int, default=250,
                    help="save a resumable checkpoint every N steps (crash-resume)")
    args = ap.parse_args()

    if not (Path(args.data) / "metadata.jsonl").exists():
        raise SystemExit(f"no dataset at {args.data} — run scripts/prepare_style_dataset.py first")

    trainer = ensure_trainer()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # 6 GB budget: fp16 + gradient checkpointing + 8-bit Adam + batch 1.
    # xformers is intentionally omitted (fragile on py3.13); torch-2.6 SDPA is enough.
    cmd = [
        "accelerate", "launch", "--mixed_precision=fp16", str(trainer),
        f"--pretrained_model_name_or_path={STYLE_BASE_MODEL}",
        f"--train_data_dir={args.data}", "--caption_column=text",
        f"--resolution={args.resolution}", "--center_crop",
        "--train_batch_size=1", f"--gradient_accumulation_steps={args.grad_accum}",
        "--gradient_checkpointing", "--use_8bit_adam",
        "--mixed_precision=fp16", f"--rank={args.rank}",
        f"--max_train_steps={args.steps}", f"--learning_rate={args.lr}",
        "--lr_scheduler=constant", "--lr_warmup_steps=0",
        "--dataloader_num_workers=2", "--seed=0",
        f"--checkpointing_steps={args.checkpointing_steps}", "--checkpoints_total_limit=2",
        "--resume_from_checkpoint=latest",
        f"--output_dir={args.output}",
    ]
    print("launching:\n  " + " ".join(cmd) + "\n")
    # expandable_segments curbs the VRAM fragmentation that triggers transient
    # "CUDA error: unknown error" on WSL2 during sustained training.
    env = {**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
