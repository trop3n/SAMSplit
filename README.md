# SAMSplit

Turn flat, painterly artwork (atmospheric digital paintings) into **layered, animatable assets
for After Effects** — by interactively segmenting elements, inpainting the paint
*behind* them, ordering them by depth, and exporting a one-click AE import.

## The problem it solves

Animating a flat painting breaks on three things:

1. **Soft edges** — painterly work has no crisp object boundary; a tree dissolves
   into sky. Hard masks look wrong, so we keep a feathered alpha.
2. **Dis-occlusion** — the instant you separate an element and move it, you expose an
   *unpainted hole* where it used to be. Something has to invent plausible paint.
3. **Depth / order** — believable parallax needs to know what's near vs. far.

SAMSplit automates all three and hands the motion designer clean, pre-positioned layers.

## Pipeline

```
PNG
 → SAM 2            interactive point/box segmentation
 → soft alpha matte feathered edges that match painterly boundaries
 → LaMa inpaint     fill the paint behind each layer / a clean background plate
 → Depth Anything V2 order layers back-to-front
 → export           full-canvas PNG layers + manifest.json + import_to_AE.jsx
```

## Modules

- `samsplit/segment.py` — SAM 2 point/box prompting → masks
- `samsplit/matte.py`   — binary mask → feathered alpha (keeps painterly edges)
- `samsplit/inpaint.py` — LaMa fill behind each layer / background plate
- `samsplit/style.py`   — Phase 4: artist-LoRA style refine of the dis-occlusion fill
- `samsplit/depth.py`   — Depth Anything V2 → back-to-front order
- `samsplit/export.py`  — PNG layers + manifest + import_to_AE.jsx (flat or 2.5D) + layered `.psd` + flattened composite (`.png`/`.jpg`)
- `samsplit/pipeline.py`— orchestration
- `app.py`              — Gradio interactive UI

## After Effects handoff

Every layer is exported as a **full-canvas transparent PNG** — the element sits in its
true position on a transparent full-size canvas — so layers stack with **zero offset**.
`import_to_AE.jsx` builds a comp at the source resolution, imports all layers, and
stacks them in depth order. One click for the motion designer.

## Roadmap

- **Phase 1 ✅** — interactive segmentation + full-canvas layer export + one-click AE import
- **Phase 2 ✅** — dis-occlusion inpainting (LaMa clean background plate; inpaint toggle, halo-dilate slider, plate preview)
- **Edge quality ✅** — ViTMatte alpha matting for soft painterly edges (per-element *Edge mode*: Matting/Feather + *Detail band* slider)
- **Phase 3 ✅** — depth-based auto-ordering + 2.5D parallax rig (Depth Anything V2 → per-layer Z + camera in the AE script; `depth.png` export; depth preview + reverse-depth toggle)
- **Phase 4 ✅** — style-matched fill: hybrid LaMa + **artist-LoRA** refine of the dis-occlusion hole. LoRA trained on the artist's catalog (~2 GB on the 3050); **clear win on large holes / textured regions**, subtle on small smooth fills — an off-by-default toggle with a strength slider (see below).

## Setup

Requires Python >= 3.10 and (recommended) an NVIDIA GPU. Uses [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync          # installs deps incl. the CUDA build of PyTorch (~2.7 GB)
uv run app.py    # launches the Gradio UI
```

Model weights (~0.8 GB) auto-download to `~/.cache/huggingface` on first use.

## Style-matched fill (Phase 4)

The dis-occlusion fill (Phase 2) uses generic LaMa. Phase 4 adds an optional second
pass that re-paints the filled hole in the *artist's own style* with a LoRA, while
keeping LaMa's structure (low-strength SD 1.5 img2img, composited back into the hole
only). The diffusion model is resident only during a refine and peaks ~2 GB on a 6 GB card.

The **"Style-match fill (artist LoRA)"** toggle in the *Background fill* panel stays
disabled until a LoRA exists at `models/style_lora/*.safetensors`. To train one on
~20–50 of the artist's images, locally on the 3050:

```bash
uv sync --extra train                                   # bitsandbytes (8-bit Adam) + datasets + tensorboard
uv run python scripts/prepare_style_dataset.py /path/to/artist_images
# close other GPU apps (incl. the SAMSplit app) — training needs all the VRAM
uv run python scripts/train_style_lora.py               # 512 px, rank 16, 1500 steps
# if it OOMs:  --resolution 448  (then 384),  or  --rank 8
```

Output lands at `models/style_lora/pytorch_lora_weights.safetensors`, auto-detected on
next launch (the toggle activates). The `STYLE_TRIGGER` token in `samsplit/config.py`
is baked into both the training captions and the refine prompt — keep them in lockstep.
If local training won't fit, the *same* script + flags run unchanged on a rented
GPU/Colab and produce the same file.

## Environment notes (this machine)

RTX 3050 6 GB (WSL2 CUDA), 20 CPU, 7.6 GB RAM, Ubuntu 24.04. With 6 GB VRAM, use
`base`/`small` models and keep one resident at a time (`low_vram=True` in `config.py`).
