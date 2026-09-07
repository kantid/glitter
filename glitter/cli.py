"""Command-line interface."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import config, analyze, layers, render, export

#: Resolution fraction for --preview, and a floor in pixels. Without the floor
#: a preview of a small picture comes out around 150px wide, where nothing can
#: be made out — and making things out is the entire point of a preview.
PREVIEW_SCALE = 0.25
PREVIEW_MIN_WIDTH = 420


def _load_image(path: Path, width: int | None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if width and width < im.width:
        h = round(im.height * width / im.width)
        im = im.resize((width, h), Image.LANCZOS)
    return np.asarray(im)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="glitter",
        description="Turn any picture into a gloriously tacky sparkling card.",
    )
    p.add_argument("image", type=Path, nargs="?")
    p.add_argument("--list-presets", action="store_true")
    p.add_argument(
        "-o", "--out", type=Path,
        help="output base name (defaults to the image name)",
    )
    p.add_argument("--preset", default="default")
    p.add_argument("--seed", type=int, help="override the preset's seed")
    p.add_argument("--frames", type=int, help="override the cycle length")
    p.add_argument("--fps", type=int)
    p.add_argument("--width", type=int, help="resize the source before rendering")
    p.add_argument(
        "--protect",
        type=Path,
        help="exclusion mask: white means leave free of glitter",
    )
    p.add_argument(
        "--dump-density",
        action="store_true",
        help="save the density map as PNG, for tuning the [density] weights",
    )
    p.add_argument(
        "--gif-width", type=int, default=560,
        help="GIF width (0 to skip resizing)",
    )
    p.add_argument("--gif-colors", type=int, default=64)
    p.add_argument(
        "--no-delta", action="store_true",
        help="GIF without delta encoding (roughly twice the size)",
    )
    p.add_argument(
        "--webp-width", type=int, default=900,
        help="WebP width (0 to skip resizing)",
    )
    p.add_argument("--webp-quality", type=int, default=70)
    p.add_argument(
        "--preview",
        action="store_true",
        help="one frame at 25%% resolution as PNG, for tuning parameters",
    )
    p.add_argument(
        "--flash-extreme",
        action="store_true",
        help="full-frame white strobe instead of a gentle brightness lift. "
        "Flashing is dangerous for photosensitive viewers — see the README",
    )
    p.add_argument("--no-gif", action="store_true")
    p.add_argument("--no-webp", action="store_true")
    args = p.parse_args(argv)

    if args.list_presets:
        for name in config.available():
            cfg = config.load(name)
            total = sum(s.count for s in cfg.systems)
            flash = "none" if not cfg.flash.strength else f"x{1 + cfg.flash.strength:.2f}"
            print(
                f"{name:20s} {total:6d} particles/Mpx, {len(cfg.sweeps)} sweeps, "
                f"{cfg.frames} frames, flash {flash}"
            )
        return 0
    if args.image is None:
        p.error("an image path is required")

    cfg = config.load(args.preset)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.frames is not None:
        cfg.frames = args.frames
    if args.fps is not None:
        cfg.fps = args.fps
    if args.flash_extreme:
        cfg.flash.strength = max(cfg.flash.strength, 0.9)
        cfg.flash.white = max(cfg.flash.white, 0.55)
        print("strobe flash enabled — flashing is dangerous for photosensitive viewers")

    width = args.width
    if args.preview:
        cfg.frames = 1
        with Image.open(args.image) as probe:
            target = max(PREVIEW_MIN_WIDTH, round(probe.width * PREVIEW_SCALE))
            width = min(width or probe.width, target)

    stem = args.out or args.image.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    rgb = _load_image(args.image, width)
    print(f"source {rgb.shape[1]}x{rgb.shape[0]}")

    protect = None
    if args.protect:
        mask = Image.open(args.protect).convert("L").resize(
            (rgb.shape[1], rgb.shape[0]), Image.LANCZOS
        )
        protect = np.asarray(mask).astype(np.float32) / 255.0

    density = analyze.density_map(rgb.astype(np.float32) / 255.0, cfg.density, protect)
    if args.dump_density:
        out = Path(f"{stem}_density.png")
        Image.fromarray((density * 255).astype(np.uint8), "L").save(out)
        print(f"-> {out}")

    bk = layers.build(rgb, density, cfg)
    t_build = time.perf_counter() - t0
    total = sum(layers.particle_count(s, *rgb.shape[:2]) for s in cfg.systems)
    print(f"{len(bk)} buckets, {total} particles, layers built in {t_build:.1f}s")

    sweeps = layers.Sweeps(cfg.sweeps, *rgb.shape[:2])
    t1 = time.perf_counter()
    frames = render.render(rgb, bk, cfg, sweeps)
    t_render = time.perf_counter() - t1
    print(
        f"{len(frames)} frames in {t_render:.1f}s "
        f"({t_render / len(frames) * 1000:.0f} ms/frame)"
    )

    if args.preview:
        out = Path(f"{stem}_preview.png")
        Image.fromarray(frames[0], "RGB").save(out)
        print(f"-> {out}")
        return 0

    seam = render.loop_seam_error(frames)
    print(f"loop seam: {seam:.2f} (1.0 is ideal)")

    if not args.no_gif:
        out = Path(f"{stem}_glitter.gif")
        export.save_gif(frames, out, cfg.fps, args.gif_width or None,
                        args.gif_colors, cfg.seed, delta=not args.no_delta)
        print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if not args.no_webp:
        out = Path(f"{stem}_glitter.webp")
        export.save_webp(frames, out, cfg.fps, args.webp_width or None,
                         args.webp_quality)
        print(f"-> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
