"""Phase 4 step B: prepare the artist's catalog as a LoRA training set.

    uv run python scripts/prepare_style_dataset.py /path/to/artist_images
    uv run python scripts/prepare_style_dataset.py ./ArtTraining --out data/style_dataset

Slices each painting into overlapping square tiles along its long axis (each resized
to `size`) and writes an imagefolder dataset (tiles + metadata.jsonl) that
scripts/train_style_lora.py feeds to the diffusers LoRA trainer. Tiling matters here:
the catalog is wide title art (~2.2:1), so a single center crop would discard most of
each painting's width — tiling keeps the whole brushwork and multiplies a small
catalog into more style tiles. Every caption carries the STYLE_TRIGGER token, so the
learned style activates whenever that token appears in the refine prompt at inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image

from samsplit.config import ROOT, STYLE_TRIGGER

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def square_tiles(img: Image.Image, size: int, max_tiles: int = 4) -> list[Image.Image]:
    """Slice into overlapping square tiles along the long axis, each resized to (size, size).

    A near-square image yields one (center) tile; a 2.2:1 image yields ~3 overlapping
    tiles spanning its width.
    """
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    span = max(w, h)
    n = min(max_tiles, max(1, math.ceil(span / s)))
    offs = [(span - s) // 2] if n == 1 else [round(i * (span - s) / (n - 1)) for i in range(n)]
    tiles = []
    for off in offs:
        box = (off, 0, off + s, s) if w >= h else (0, off, s, off + s)
        tiles.append(img.crop(box).resize((size, size), Image.LANCZOS))
    return tiles


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a style-LoRA dataset from the artist's images.")
    ap.add_argument("src", help="folder of the artist's images")
    ap.add_argument("--out", default=str(ROOT / "data" / "style_dataset"))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max-tiles", type=int, default=4, help="max square tiles per image along its long axis")
    ap.add_argument("--caption", default=f"a painting in {STYLE_TRIGGER} style")
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    srcs = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS) if src.is_dir() else []
    if not srcs:
        raise SystemExit(f"no images found in {src}")

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(srcs):
        for j, tile in enumerate(square_tiles(Image.open(p), args.size, args.max_tiles)):
            name = f"{i:03d}_{j}.png"
            tile.save(out / name)
            rows.append({"file_name": name, "text": args.caption})
    (out / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    print(f"wrote {len(rows)} tiles from {len(srcs)} images + metadata.jsonl -> {out}")
    print(f"caption: {args.caption!r}")
    if len(rows) < 15:
        print(f"NOTE: {len(rows)} tiles is low for a style LoRA (~20-50+ is the sweet spot).")


if __name__ == "__main__":
    main()
