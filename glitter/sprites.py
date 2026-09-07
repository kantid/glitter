"""Sprite atlas.

Sprites are pre-rendered once and only indexed afterwards — rotating and
scaling on the fly is forbidden, and those two are exactly what would cost ten
minutes per render. So rotations are discrete: the atlas holds every size at
every angle, and a particle simply draws a random index.

A ray is drawn from the distance to its axis rather than from an angular lobe
of the form |cos(n*theta/2)|^p. A lobe has constant angular width, which means
in pixels it widens linearly with radius and turns into a wedge at the tip.
What is needed is a needle.
"""

from __future__ import annotations

import numpy as np

#: Discrete angles per size, spanning one symmetry sector of the star — so for
#: a four-ray star that is 8 angles across 90 degrees.
ROTATIONS = 8

#: Supersampling factor. A sprite is evaluated on a grid SUPERSAMPLE times
#: finer and averaged over blocks.
#:
#: Without this a 3px dot looks like a diamond, and it cannot look like
#: anything else: centre 1.0, four sides 0.54, four corners 0.29 — you cannot
#: draw a circle on nine pixels. It has to be computed at high resolution and
#: brought back down.
SUPERSAMPLE = 4


def _grid(size: int, ss: int = SUPERSAMPLE) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subpixel coordinates: (size*ss, size*ss), centred on the sprite centre."""
    r = size // 2
    offsets = (np.arange(ss, dtype=np.float32) + 0.5) / ss - 0.5
    axis = (np.arange(-r, r + 1, dtype=np.float32)[:, None] + offsets).ravel()
    x, y = np.meshgrid(axis, axis)
    return x, y, np.hypot(x, y)


def _downsample(a: np.ndarray, size: int, ss: int = SUPERSAMPLE) -> np.ndarray:
    return a.reshape(size, ss, size, ss).mean(axis=(1, 3)).astype(np.float32)


def dot(size: int, rotation: float = 0.0) -> np.ndarray:
    """Gaussian dot, float32 alpha in [0, 1], shape (size, size)."""
    _, _, dist = _grid(size)
    # A sigma of 0.30 cut the gaussian off at 25% brightness at the rim — a
    # hard edge, and at small sizes that edge is what reads as a polygon. At
    # 0.22 only 8% is left at the rim and the cutoff is barely visible.
    sigma = max(size * 0.22, 0.5)
    a = np.exp(-(dist**2) / (2.0 * sigma * sigma))
    a[dist > size / 2.0] = 0.0
    return _downsample(a, size)


def star(size: int, rotation: float = 0.0, rays: int = 4,
         alternate: bool = False) -> np.ndarray:
    """Star with rays: core, soft bloom, needles.

    Args:
        rays: how many visible rays.
        alternate: shorten every other ray — the classic look, four long rays
            on the axes and four short ones on the diagonals.
    """
    x, y, dist = _grid(size)
    r = size // 2
    width = max(size * 0.035, 0.55)

    spikes = np.zeros_like(dist)
    for i in range(rays):
        phi = rotation + 2.0 * np.pi * i / rays
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        along = x * cos_p + y * sin_p
        perp = -x * sin_p + y * cos_p
        length = r * (0.5 if (alternate and i % 2) else 1.0)
        # A linear falloff along the ray gives a crisp tip, the gaussian across
        # it a soft edge. An exponent below one keeps the ray bright almost to
        # the end: at 1.5 the middle dropped to 35% and the star read as a blob.
        taper = np.clip(1.0 - along / max(length, 1e-3), 0.0, 1.0) ** 0.9
        ray = np.where(along >= 0.0, np.exp(-((perp / width) ** 2)) * taper, 0.0)
        spikes = np.maximum(spikes, ray)

    # The bloom is kept tight: at a sigma of 0.22 of the size it swallowed the
    # short rays of star8, making the eight-ray star indistinguishable from the
    # four-ray one.
    core = np.exp(-((dist / max(size * 0.05, 0.5)) ** 2))
    bloom = 0.25 * np.exp(-((dist / max(size * 0.10, 0.5)) ** 2))

    a = np.clip(spikes + core + bloom, 0.0, 1.0)
    a[dist > r + 0.5] = 0.0
    return _downsample(a, size)


def star4(size: int, rotation: float = 0.0) -> np.ndarray:
    return star(size, rotation, rays=4)


def star6(size: int, rotation: float = 0.0) -> np.ndarray:
    return star(size, rotation, rays=6)


def star8(size: int, rotation: float = 0.0) -> np.ndarray:
    return star(size, rotation, rays=8, alternate=True)


def speck(size: int, rotation: float = 0.0) -> np.ndarray:
    """Hard aliased pixel — the authentic spark of the era.

    Glitter in 2000s cards was made by flood-filling with a noise tile in an
    indexed palette: a pixel is either fully lit or not. There were no gaussian
    midtones to be had. So antialiasing is deliberately off here, unlike `dot`.
    """
    return np.ones((size, size), dtype=np.float32)


BUILDERS = {"dot": dot, "speck": speck, "star4": star4, "star6": star6, "star8": star8}

#: Symmetry sector: past this angle a rotation repeats an already built sprite.
#: Rotation is meaningless for a dot and a speck; star8 with its alternation has
#: twice the period, so it shares the four-ray sector.
SECTORS = {"dot": 1, "speck": 1, "star4": 4, "star6": 6, "star8": 4}

#: Sprites that do not need odd sizes: a hard pixel has no centre to preserve,
#: and size 2 shows up constantly in the tiles of that era.
ANY_SIZE = {"speck"}


def build_atlas(kind: str, size_range: tuple[int, int]) -> list[np.ndarray]:
    """Every size in the range, at every angle.

    Sizes are odd so that a sprite has exactly one centre pixel — except for
    the sprites listed in ANY_SIZE.
    """
    if kind not in BUILDERS:
        raise SystemExit(f"unknown sprite: {kind} (available: {', '.join(BUILDERS)})")

    lo, hi = size_range
    if kind in ANY_SIZE:
        sizes = list(range(max(1, lo), max(1, hi) + 1))
    else:
        sizes = [s for s in range(max(3, lo), hi + 1) if s % 2 == 1]
    if not sizes:
        sizes = [max(1, lo) if kind in ANY_SIZE else max(3, lo | 1)]

    sector = SECTORS[kind]
    n_rot = 1 if sector == 1 else ROTATIONS
    build = BUILDERS[kind]
    return [
        build(s, 2.0 * np.pi * i / (sector * n_rot))
        for s in sizes
        for i in range(n_rot)
    ]
