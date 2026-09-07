"""Export: quantisation, GIF, animated WebP.

Quantisation is not a formality at the end of the pipeline, it is half of how
the result looks. Three rules, the first two of which are why home-made
glitter GIFs usually look muddy:

1. The palette is built from a sample of ALL frames, not the first one.
   Otherwise the colours of sparks absent from frame one get mapped to
   whatever neighbour is closest.
2. dither=NONE. Dithering smears point highlights into noise; 128 colours
   without it look better than 256 with it.
3. Delta encoding through transparency. It seemed at first that with
   everything twinkling every pixel changes and there is nothing to save —
   measurement showed the opposite: about 72% of pixels match between adjacent
   frames, wherever there is simply no glitter. Marking those with a
   transparent index and setting disposal=1 gives LZW long runs of the same
   index, which it compresses very well.

Pillow's own optimize=True really is useless here: all it does is crop a frame
to the bounding box of the changes, and glitter is spread across the whole area.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

#: How many pixels to take from each frame when building the global palette.
PALETTE_SAMPLE_PER_FRAME = 20_000


def global_palette(frames: list[np.ndarray], colors: int, seed: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    chunks = []
    for f in frames:
        flat = f.reshape(-1, 3)
        idx = rng.integers(0, flat.shape[0], PALETTE_SAMPLE_PER_FRAME)
        chunks.append(flat[idx])
    sample = np.concatenate(chunks).reshape(-1, 1, 3)
    return Image.fromarray(sample, "RGB").quantize(
        colors=colors, method=Image.Quantize.MEDIANCUT
    )


def _resized(frames: list[np.ndarray], width: int | None) -> list[Image.Image]:
    images = [Image.fromarray(f, "RGB") for f in frames]
    if width and width < images[0].width:
        h = round(images[0].height * width / images[0].width)
        images = [im.resize((width, h), Image.LANCZOS) for im in images]
    return images


def save_gif(frames: list[np.ndarray], path: Path, fps: int,
             width: int | None = None, colors: int = 128, seed: int = 0,
             delta: bool = True) -> None:
    images = _resized(frames, width)
    # One index is reserved for transparency, the rest go to the palette.
    n_colors = colors - 1 if delta else colors
    transparent = n_colors
    pal = global_palette([np.asarray(im) for im in images], n_colors, seed)
    quantized = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in images]

    if delta:
        prev = np.asarray(quantized[0])
        deltas = [quantized[0]]
        for im in quantized[1:]:
            cur = np.asarray(im)
            masked = np.where(cur == prev, transparent, cur).astype(np.uint8)
            out = Image.fromarray(masked, "P")
            out.putpalette(quantized[0].getpalette())
            deltas.append(out)
            prev = cur
        quantized = deltas

    extra = {"transparency": transparent} if delta else {}
    quantized[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=quantized[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
        disposal=1,
        **extra,
    )


def save_webp(frames: list[np.ndarray], path: Path, fps: int,
              width: int | None = None, quality: int = 80) -> None:
    images = _resized(frames, width)
    images[0].save(
        path,
        format="WEBP",
        minimize_size=True,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
        quality=quality,
        method=4,
    )
