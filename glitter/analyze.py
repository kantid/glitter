"""Glitter density maps.

Without a neural network what you extract from a picture is statistics, not
semantics, so there is no `flower_mask` or `text_mask` here in the sense of
"we found a flower". There is a weighted sum of features, and particle
positions are sampled from it:

    density = luma^gamma + saturation + edges + text + sum(hue_bands) - protect

What that buys in practice (see the table in PLAN.md): roses are caught by a
hue band around magenta, gold by a band around yellow, white lettering by a
top-hat transform (bright strokes thinner than the structuring element), and
faces are protected by an HSV skin heuristic or an explicit --protect mask.

Every feature is normalised by percentile rather than min/max: one stray
blown-out highlight would otherwise crush the rest of the picture to zero.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import DensitySpec

#: Percentiles used to normalise a feature into [0, 1].
CLIP_PCT = (1.0, 99.0)


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, CLIP_PCT)
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """H in degrees [0, 360), S and V in [0, 1]. `rgb` is float32 in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    d = mx - mn
    safe_d = np.where(d == 0, 1.0, d)

    h = np.select(
        [mx == r, mx == g],
        [(g - b) / safe_d % 6.0, (b - r) / safe_d + 2.0],
        default=(r - g) / safe_d + 4.0,
    )
    h = np.where(d == 0, 0.0, h * 60.0)
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0.0)
    return h.astype(np.float32), s.astype(np.float32), mx.astype(np.float32)


def luma(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]).astype(
        np.float32
    )


def edge_energy(gray: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Blurred gradient magnitude.

    Blurred on purpose: glitter should settle NEAR a contour, not exactly on it.
    """
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    return _norm(ndimage.gaussian_filter(np.hypot(gx, gy), sigma))


def hue_band(h: np.ndarray, s: np.ndarray, center: float, width: float) -> np.ndarray:
    """Gaussian band around a hue, weighted by saturation.

    Distance wraps around the circle: hues 350 and 10 are 20 degrees apart, not
    340. Multiplying by saturation is mandatory — the hue of a grey pixel is
    meaningless.
    """
    d = np.abs(h - center)
    d = np.minimum(d, 360.0 - d)
    return (np.exp(-((d / max(width, 1e-3)) ** 2)) * s).astype(np.float32)


def text_mask(gray: np.ndarray, stroke: int) -> np.ndarray:
    """White top-hat: bright structures thinner than `stroke` pixels.

    This is what catches the white outline of lettering and thin highlights
    while ignoring large bright areas such as sky or a dress.
    """
    opened = ndimage.grey_opening(gray, size=(stroke, stroke))
    return _norm(np.maximum(gray - opened, 0.0))


def skin_mask(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Coarse HSV skin-tone heuristic.

    Face detection without a neural network is not possible, and this is not
    it: the mask will catch any skin and some warm beige surfaces too. For
    precise protection there is --protect. The upper saturation bound keeps
    roses out, the lower brightness bound keeps shadows out.
    """
    m = (h >= 5.0) & (h <= 45.0) & (s >= 0.12) & (s <= 0.60) & (v >= 0.25)
    m = ndimage.binary_closing(m, structure=np.ones((5, 5)))
    m = ndimage.binary_opening(m, structure=np.ones((5, 5)))
    return ndimage.gaussian_filter(m.astype(np.float32), 6.0)


def density_map(
    rgb: np.ndarray, spec: DensitySpec, protect: np.ndarray | None = None
) -> np.ndarray:
    """Density map in [0, 1], shape (H, W).

    Args:
        rgb: image as float32 in [0, 1], shape (H, W, 3).
        spec: feature weights from the preset.
        protect: optional mask in [0, 1] where 1 means "leave alone".
    """
    h, s, v = hsv(rgb)
    gray = luma(rgb)

    d = np.zeros(gray.shape, dtype=np.float32)
    if spec.luma:
        d += spec.luma * _norm(gray) ** spec.luma_gamma
    if spec.saturation:
        d += spec.saturation * _norm(s)
    if spec.edges:
        d += spec.edges * edge_energy(gray)
    if spec.text:
        stroke = max(3, round(min(gray.shape) / 120) | 1)
        d += spec.text * text_mask(gray, stroke)
    for band in spec.hue_bands:
        d += band.weight * hue_band(h, s, band.center, band.width)

    # The floor is raised BEFORE protection, otherwise it cancels protection
    # out: the background should sparkle everywhere, but an explicitly
    # protected area has to reach zero rather than settle at the floor.
    d = np.clip(_norm(d), spec.floor, 1.0)

    keep = np.ones_like(d)
    if spec.protect_skin:
        keep *= 1.0 - spec.protect_skin * skin_mask(h, s, v)
    if protect is not None:
        keep *= 1.0 - np.clip(protect, 0.0, 1.0)
    return d * keep


#: Frame side the sizes in `contour_mask` are specified for. Outline thickness
#: and the structuring elements scale from it; otherwise --preview shows
#: something other than the full render, with the outline coming out twice as
#: thick and twice as dense relative to the frame.
CONTOUR_REFERENCE_SIDE = 800


def contour_mask(density: np.ndarray, threshold: float, width: int) -> np.ndarray:
    """Outline around the glittered regions.

    A direct analogue of the trick from period tutorials: the area to be
    glittered was selected, the selection grown by a pixel or two
    (`Selections > Expand`), and a hard bright border ran along the edge. That
    border is what reads as "sparkling", not the scattered dots on their own.

    `width` is a RADIUS: the outline extends `width` pixels outward from the
    region border and the same distance inward, so the ring is twice as thick.
    """
    scale = min(density.shape) / CONTOUR_REFERENCE_SIDE
    w = max(1, round(width * scale))
    close = max(3, round(5 * scale) | 1)
    open_ = max(3, round(3 * scale) | 1)

    # Morphology runs on a padded image with the border replicated outward.
    # Otherwise scipy treats everything past the edge as background, a region
    # touching the edge gets eroded from outside, and dilation & ~erosion
    # lights up an outline around the whole frame where no border exists: on a
    # pastel card that lit a quarter of all edge pixels.
    pad = w + close + open_
    solid = np.pad(density > threshold, pad, mode="edge")

    # Close the holes, or noise inside a region outlines every speckle.
    solid = ndimage.binary_closing(solid, structure=np.ones((close, close)))
    solid = ndimage.binary_opening(solid, structure=np.ones((open_, open_)))

    k = np.ones((2 * w + 1, 2 * w + 1))
    ring = ndimage.binary_dilation(solid, k) & ~ndimage.binary_erosion(solid, k)
    return ring[pad:-pad, pad:-pad]


def sample_positions(
    density: np.ndarray, n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """n positions distributed proportionally to density. Returns (ys, xs)."""
    flat = density.astype(np.float64).ravel()
    total = flat.sum()
    if total <= 0:
        raise ValueError("density map is empty")

    # cumsum + searchsorted rather than rng.choice(p=...): markedly faster
    # across 1.5M pixels.
    cdf = np.cumsum(flat)
    picks = np.searchsorted(cdf, rng.random(n) * total)
    picks = np.clip(picks, 0, flat.size - 1)
    return np.unravel_index(picks, density.shape)
