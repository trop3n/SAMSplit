"""Phase 4: style-matched dis-occlusion fill.

After LaMa fills a hole with structurally-plausible paint (``inpaint.py``),
``StyleRefiner`` runs a *light* Stable Diffusion img2img pass — driven by a LoRA
fine-tuned on the artist's own catalog — over a crop of the hole, then composites
the result back INTO THE HOLE ONLY. Low img2img strength keeps LaMa's structure and
just re-imposes the artist's brushwork and palette; the surrounding valid pixels in
the crop steer the diffusion so the texture continues seamlessly across the seam.

Lazy + VRAM-safe like the other model wrappers (``LamaInpainter``, ``ViTMatteMatter``):
the pipeline loads on first use, runs with CPU offload + attention/VAE slicing, caps
the crop at ``max_side``, and retries at lower resolution / fewer steps on OOM. On a
6 GB card it is only resident during a refine call.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from samsplit.matte import soft_alpha_from_mask


class StyleRefiner:
    """Lazy SD 1.5 img2img + artist LoRA; re-textures an already-filled hole."""

    def __init__(self, base_id: str, lora_path: str | Path | None, device: str | None = None,
                 *, max_side: int = 768, prompt: str = "", negative_prompt: str = "",
                 guidance: float = 5.0, seed: int = 0):
        self.base_id = base_id
        self.lora_path = lora_path  # dir or .safetensors file, or None (base style only)
        self._device = device
        self.max_side = max_side
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.guidance = guidance
        self.seed = seed
        self._pipe = None

    def _model(self):
        if self._pipe is None:
            import torch
            from diffusers import StableDiffusionImg2ImgPipeline

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.base_id, torch_dtype=dtype,
                safety_checker=None, requires_safety_checker=False,
            )
            # Load the artist LoRA BEFORE attaching offload hooks. load_lora_weights
            # accepts a directory (auto-finds the weights file) or an explicit file.
            if self.lora_path is not None:
                p = Path(self.lora_path)
                if p.is_file():
                    pipe.load_lora_weights(str(p.parent), weight_name=p.name)
                else:
                    pipe.load_lora_weights(str(p))
            pipe.set_progress_bar_config(disable=True)
            if self._device == "cuda":
                # Keep peak < 6 GB: stream modules on/off the GPU + slice attention/VAE.
                pipe.enable_model_cpu_offload()
                pipe.enable_attention_slicing()
                pipe.vae.enable_slicing()  # pipe.enable_vae_slicing() is deprecated
            else:
                pipe.to(self._device)
            self._pipe = pipe
        return self._pipe

    def refine(self, image_rgb: np.ndarray, hole_mask: np.ndarray, *, dilate_px: int = 4,
               strength: float = 0.3, steps: int = 15, prompt: str | None = None,
               seed: int | None = None, pad: int = 32) -> np.ndarray:
        """Re-texture the hole region of an already-filled plate with the artist LoRA.

        ``image_rgb``: (H,W,3) uint8 — the LaMa-filled plate.
        ``hole_mask``: (H,W) bool/uint8 — True where paint was invented (the hole).
        Returns (H,W,3) uint8 with only the hole region restyled (feathered composite);
        pixels outside the (dilated, feathered) hole are preserved.
        """
        rgb = np.ascontiguousarray(image_rgb[:, :, :3]).astype(np.uint8)
        h, w = rgb.shape[:2]
        hole = (hole_mask > 0).astype(np.uint8)
        if dilate_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
            hole = cv2.dilate(hole, k)
        if int(hole.sum()) == 0:
            return rgb  # nothing was filled

        # Restyle only locally: crop to the hole bbox + padding (context for the diffusion).
        ys, xs = np.where(hole > 0)
        y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + 1 + pad)
        x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + 1 + pad)
        crop = rgb[y0:y1, x0:x1]
        crop_hole = hole[y0:y1, x0:x1]

        styled = self._run_img2img(crop, strength=strength, steps=steps,
                                   prompt=prompt or self.prompt,
                                   seed=self.seed if seed is None else seed)

        # Feathered composite: keep the restyled paint only inside the hole so the
        # seam against LaMa's surrounding fill is soft (no hard rectangle edge).
        feather = max(1.0, dilate_px / 2.0)
        a = soft_alpha_from_mask(crop_hole, feather=feather, erode_px=0)[..., None]
        blended = styled.astype(np.float32) * a + crop.astype(np.float32) * (1.0 - a)
        out = rgb.copy()
        out[y0:y1, x0:x1] = blended.clip(0, 255).astype(np.uint8)
        return out

    def _run_img2img(self, crop_rgb: np.ndarray, *, strength: float, steps: int,
                     prompt: str, seed: int) -> np.ndarray:
        import torch

        pipe = self._model()
        ch, cw = crop_rgb.shape[:2]
        max_side = self.max_side
        cur_steps = steps
        for _ in range(3):
            scale = min(1.0, max_side / max(ch, cw))
            cw2, ch2 = int(round(cw * scale)), int(round(ch * scale))
            cw2, ch2 = max(8, cw2 - cw2 % 8), max(8, ch2 - ch2 % 8)  # SD needs /8 dims
            img_s = cv2.resize(crop_rgb, (cw2, ch2), interpolation=cv2.INTER_AREA)
            try:
                gen = torch.Generator(device="cpu").manual_seed(int(seed))
                with torch.inference_mode():
                    res = pipe(
                        prompt=prompt,
                        negative_prompt=self.negative_prompt or None,
                        image=Image.fromarray(img_s),
                        strength=float(strength),
                        num_inference_steps=int(cur_steps),
                        guidance_scale=float(self.guidance),
                        generator=gen,
                    ).images[0]
                out = np.asarray(res.convert("RGB"))
                if out.shape[:2] != (ch, cw):
                    out = cv2.resize(out, (cw, ch), interpolation=cv2.INTER_LINEAR)
                return out
            except RuntimeError as err:
                if "out of memory" in str(err).lower() and max_side > 384:
                    self._empty_cache()
                    max_side = max(384, max_side // 2)
                    cur_steps = max(8, cur_steps // 2)
                    continue
                raise
        raise RuntimeError("Style refine ran out of memory even at reduced resolution")

    @staticmethod
    def _empty_cache():
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
