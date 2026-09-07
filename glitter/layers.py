"""Phase buckets — the core idea of the whole thing.

Glitter is glued to the surface: the particles do not move, only their
brightness changes. So instead of 32,000 draw calls per frame we render K
static layers once and add them up each frame with sinusoidal weights:

    frame(t) = base + sum_k w_k(t) * layer_k

A bucket is a pair (harmonic m, phase slot p). The harmonic sets how fast that
group twinkles, the phase slot offsets it so the particles do not blink in
unison. Only integer m is allowed: w_m = 2*pi*m/frames, otherwise the loop
tears at the seam.
"""

from __future__ import annotations

import numpy as np

from .config import Config, FlashSpec, GradeSpec, SweepSpec, SystemSpec, hex_to_rgb
from .analyze import contour_mask, sample_positions
from .sprites import build_atlas


class Buckets:
    """K static layers plus their weights as a function of the frame."""

    def __init__(self, layers: np.ndarray, harmonics: np.ndarray, phases: np.ndarray):
        #: (K, H, W, 3) uint8 — as float32 the same layers would cost 4x more.
        self.layers = layers
        self.harmonics = harmonics  # (K,) int
        self.phases = phases  # (K,) float, radians

    def __len__(self) -> int:
        return self.layers.shape[0]

    def weights(self, frame: int, total_frames: int) -> np.ndarray:
        """Weights of every bucket at `frame`. Shape (K,), values in [0, 1]."""
        t = frame / total_frames  # fraction of the cycle
        return 0.5 + 0.5 * np.sin(self.phases + 2.0 * np.pi * self.harmonics * t)


#: `count` in presets is per megapixel. Otherwise --preview lies: at quarter
#: resolution the area is 16x smaller, and the same absolute particle count
#: turns the picture into solid white mush.
REFERENCE_PIXELS = 1_000_000


def particle_count(spec: SystemSpec, height: int, width: int) -> int:
    return max(1, round(spec.count * height * width / REFERENCE_PIXELS))


#: Luminance weights that saturation rotates around.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)

#: Midpoint that contrast pivots around.
PIVOT = 128.0


def _luma(rgb: np.ndarray) -> np.ndarray:
    """Luminance in [0, 1] from a uint8 image."""
    return (rgb.astype(np.float32) / 255.0) @ LUMA


def _hue_matrix(degrees: float) -> np.ndarray:
    """Colour rotation about the grey axis. Grey stays grey at any angle."""
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    third = 1.0 / 3.0
    off = np.sqrt(third) * s
    d = third * (1.0 - c)
    return np.array(
        [
            [c + d, d - off, d + off],
            [d + off, c + d, d - off],
            [d - off, d + off, c + d],
        ],
        dtype=np.float32,
    )


def flash_pulse(spec: FlashSpec, frame: int, total_frames: int) -> float:
    """Triangular flash pulse, exactly one per cycle."""
    if spec.frames <= 0:
        return 0.0
    d = (frame / total_frames - spec.at) % 1.0
    d = min(d, 1.0 - d)  # circular distance: the cycle wraps
    half = max(spec.frames / 2.0 / total_frames, 1e-6)
    return float(np.clip(1.0 - d / half, 0.0, 1.0))


def grade_transform(
    grade: GradeSpec, flash: FlashSpec, frame: int, total_frames: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """3x3 matrix and offset for this frame's grade, or None if it is identity.

    Saturation, contrast, hue shift and the flash are all linear, so they fold
    into a single transform applied with one matmul instead of four passes over
    the frame.
    """
    t = frame / total_frames
    w = 2.0 * np.pi * grade.harmonic * t
    # Phases are spread a third of a period apart: in lockstep, saturation,
    # contrast and hue read as one shared brightness pulse and the point of
    # having three separate knobs is lost.
    sat = 1.0 + grade.saturation * np.sin(w)
    con = 1.0 + grade.contrast * np.sin(w + 2.0 * np.pi / 3.0)
    hue = grade.hue * np.sin(w + 4.0 * np.pi / 3.0)

    pulse = flash_pulse(flash, frame, total_frames)
    gain = 1.0 + flash.strength * pulse
    white = flash.white * pulse * 255.0

    if sat == 1.0 and con == 1.0 and hue == 0.0 and gain == 1.0 and white == 0.0:
        return None

    m_sat = np.eye(3, dtype=np.float32) * sat + (1.0 - sat) * np.outer(
        np.ones(3, dtype=np.float32), LUMA
    )
    m = gain * con * (_hue_matrix(hue) @ m_sat)
    b = np.full(3, gain * PIVOT * (1.0 - con) + white, dtype=np.float32)
    return m.astype(np.float32), b


#: Resolution of the 1-D sweep profile table. 4096 steps across the whole frame
#: is finer than a pixel, so it is indistinguishable from the exact computation.
SWEEP_LUT = 4096


class Sweeps:
    """Travelling bands of light.

    A sweep's brightness depends only on the pixel's projection onto one axis,
    that is on a single scalar. So there is no need to evaluate full-frame
    trigonometry: each frame fills a 4096-entry table, and the frame itself is
    an index lookup through a precomputed map. Without this, three sweeps would
    have doubled the per-frame cost.
    """

    def __init__(self, specs: list[SweepSpec], height: int, width: int):
        self.specs = specs
        self._axis = np.linspace(0.0, 1.0, SWEEP_LUT, dtype=np.float32)
        self._index = []
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        for spec in specs:
            a = np.deg2rad(spec.angle)
            u = x * np.cos(a) + y * np.sin(a)
            u -= u.min()
            u /= max(u.max(), 1e-6)
            self._index.append((u * (SWEEP_LUT - 1)).astype(np.uint16))

    def __bool__(self) -> bool:
        return bool(self.specs)

    def field(self, frame: int, total_frames: int) -> np.ndarray:
        """Illumination map (H, W) float32. Zero means no sweep here."""
        t = frame / total_frames
        out = None
        for spec, index in zip(self.specs, self._index):
            # Integer `passes` guarantees the phase advances by a whole number
            # of periods per cycle, so frame 0 matches frame N exactly.
            p = np.mod(self._axis * spec.bands - spec.passes * t, 1.0)
            d = np.minimum(p, 1.0 - p)  # distance to the nearest band centre
            lut = spec.strength * np.exp(-((d / max(spec.width, 1e-3)) ** 2))
            f = lut[index]
            out = f if out is None else out + f
        return out


def _stamp(layer: np.ndarray, sprite: np.ndarray, cy: int, cx: int,
           color: np.ndarray, gain: float) -> None:
    """Additively composite a sprite into a layer, clipped at the edges."""
    h, w = sprite.shape
    y0, x0 = cy - h // 2, cx - w // 2
    y1, x1 = y0 + h, x0 + w
    H, W = layer.shape[:2]

    sy0, sx0 = max(0, -y0), max(0, -x0)
    sy1, sx1 = h - max(0, y1 - H), w - max(0, x1 - W)
    if sy1 <= sy0 or sx1 <= sx0:
        return

    patch = sprite[sy0:sy1, sx0:sx1, None] * color * gain
    layer[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1] += patch


def build(rgb: np.ndarray, density: np.ndarray, cfg: Config) -> Buckets:
    """Scatter every particle system across buckets and render the layers."""
    H, W = density.shape
    rng = np.random.default_rng(cfg.seed)

    # Bucket set: every distinct harmonic x phase_slots phase offsets.
    contour_harmonics = cfg.contour.harmonics if cfg.contour.strength > 0 else ()
    all_harmonics = sorted(
        {m for s in cfg.systems for m in s.harmonics} | set(contour_harmonics)
    )
    combos = [(m, p) for m in all_harmonics for p in range(cfg.phase_slots)]
    index_of = {c: i for i, c in enumerate(combos)}
    K = len(combos)

    harmonics = np.array([m for m, _ in combos], dtype=np.int32)
    phases = np.array(
        [2.0 * np.pi * p / cfg.phase_slots for _, p in combos], dtype=np.float32
    )

    # Headroom to white at the landing spot. A speck on an already bright pixel
    # adds almost nothing — otherwise, on smooth bright areas, glitter is the
    # only high-frequency detail left and it erases the picture's structure.
    headroom = 1.0 - cfg.headroom * _luma(rgb)

    # Particles are grouped by bucket so each layer can be built in one pass:
    # only a single float32 layer is alive at a time, the rest are already uint8.
    per_bucket: list[list[tuple]] = [[] for _ in range(K)]
    for spec in cfg.systems:
        atlas = build_atlas(spec.sprite, spec.size_range)
        palette = np.array([hex_to_rgb(c) for c in spec.colors], dtype=np.float32)
        n = particle_count(spec, H, W)
        field = density if spec.density_gamma == 1.0 else density**spec.density_gamma
        ys, xs = sample_positions(field, n, rng)
        sprite_ids = rng.integers(0, len(atlas), n)
        color_ids = rng.integers(0, len(palette), n)
        ms = rng.choice(spec.harmonics, n)
        ps = rng.integers(0, cfg.phase_slots, n)
        for i in range(n):
            k = index_of[(int(ms[i]), int(ps[i]))]
            y, x = int(ys[i]), int(xs[i])
            per_bucket[k].append(
                (y, x, atlas[sprite_ids[i]], palette[color_ids[i]],
                 spec.gain * float(headroom[y, x]))
            )

    # Outline: its pixels are scattered across buckets one at a time, so it
    # shimmers instead of pulsing as one piece. Placed vectorised — there are
    # tens of thousands of outline pixels, and a per-pixel _stamp would eat the
    # entire saving the buckets bought us.
    contour_px: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (np.array([], int), np.array([], int), np.zeros((0, 3), np.float32))
    ] * K
    if cfg.contour.strength > 0:
        spec = cfg.contour
        mask = contour_mask(density, spec.threshold, spec.width)
        cys, cxs = np.nonzero(mask)
        eligible = [i for i, (m, _) in enumerate(combos) if m in spec.harmonics]
        slots = rng.choice(eligible, cys.size)
        palette = np.array([hex_to_rgb(c) for c in spec.colors], dtype=np.float32)
        cols = palette[rng.integers(0, len(palette), cys.size)] * spec.strength
        cols *= headroom[cys, cxs, None]
        contour_px = [
            (cys[slots == k], cxs[slots == k], cols[slots == k]) for k in range(K)
        ]

    layers = np.zeros((K, H, W, 3), dtype=np.uint8)
    scratch = np.zeros((H, W, 3), dtype=np.float32)
    for k, particles in enumerate(per_bucket):
        scratch[:] = 0.0
        for cy, cx, sprite, color, gain in particles:
            _stamp(scratch, sprite, cy, cx, color, gain)
        cy_k, cx_k, col_k = contour_px[k]
        if cy_k.size:
            scratch[cy_k, cx_k] += col_k
        np.clip(scratch * 255.0, 0, 255, out=scratch)
        layers[k] = scratch.astype(np.uint8)

    return Buckets(layers, harmonics, phases)
