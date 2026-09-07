"""Frame assembly from the base image and the phase buckets."""

from __future__ import annotations

import numpy as np

from .config import Config
from .layers import Buckets, Sweeps, grade_transform

#: Below this weight a bucket's contribution is invisible but still costs a
#: full pass over the array.
WEIGHT_EPS = 1.0 / 255.0

#: Ceiling of the tone table. Anything brighter ends up white regardless.
TONE_MAX = 1023


def tone_lut(knee: float) -> np.ndarray:
    """Soft highlight knee: identity below `knee`, asymptotic to 255 above it.

    Additive glitter pushes past the ceiling across a fifth of the frame, and
    hard clipping there removes colour rather than brightness: on a pink pixel
    the red channel saturates first, and the rose turns white. The knee only
    absorbs the excess, so colour around the core of a spark survives.

    Built as a table rather than evaluated per frame: `exp` over 4.7M elements
    costs more than indexing into a ready 1024-entry array.
    """
    x = np.arange(TONE_MAX + 1, dtype=np.float32)
    head = max(255.0 - knee, 1e-3)
    # Clamp the argument: at knee = 255 (knee disabled) `head` degenerates and
    # without a bound `exp` overflows.
    over = np.clip((x - knee) / head, 0.0, 60.0)
    soft = knee + head * (1.0 - np.exp(-over))
    return np.clip(np.where(x < knee, x, soft), 0, 255).astype(np.uint8)


def render(base: np.ndarray, buckets: Buckets, cfg: Config,
           sweeps: Sweeps | None = None) -> list[np.ndarray]:
    """A list of cfg.frames uint8 frames, each (H, W, 3)."""
    # Glitter accumulates separately from the base: under a sweep it has to
    # flare noticeably harder than the background, which means the two need to
    # be scalable independently.
    glitter = np.empty(base.shape, dtype=np.float32)
    acc = np.empty(base.shape, dtype=np.float32)
    tmp = np.empty(base.shape, dtype=np.float32)
    base_f = base.astype(np.float32)

    lut = tone_lut(cfg.highlight_knee)
    frames = []
    for f in range(cfg.frames):
        w = buckets.weights(f, cfg.frames)
        glitter[:] = 0.0
        for k in range(len(buckets)):
            if w[k] < WEIGHT_EPS:
                continue
            np.multiply(buckets.layers[k], w[k], out=tmp)
            glitter += tmp

        if sweeps:
            s = sweeps.field(f, cfg.frames)[..., None]
            np.multiply(base_f, 1.0 + cfg.sweep_base_gain * s, out=acc)
            np.multiply(glitter, 1.0 + cfg.sweep_glitter_gain * s, out=glitter)
            acc += glitter
        else:
            np.add(base_f, glitter, out=acc)

        # Grade before tone mapping so the knee catches its overshoot too.
        grade = grade_transform(cfg.grade, cfg.flash, f, cfg.frames)
        if grade is not None:
            m, b = grade
            # Through a separate buffer: matmul with an `out` that overlaps the
            # input is undefined. `tmp` is free again by this point.
            np.matmul(acc.reshape(-1, 3), m.T, out=tmp.reshape(-1, 3))
            np.add(tmp, b, out=acc)

        np.clip(acc, 0, TONE_MAX, out=acc)
        frames.append(lut[acc.astype(np.uint16)])
    return frames


def loop_seam_error(frames: list[np.ndarray]) -> float:
    """Difference at the loop seam against a typical transition inside it.

    A seamless loop sits near 1.0: the seam should be no different from any
    other transition. Well above 1 means a non-integer frequency crept in.

    The baseline is the median across all transitions rather than the single
    0 -> 1 step: with few phase slots and high harmonics the transitions vary
    a lot, and a coincidentally small first step inflated the ratio to 1.3 on
    a loop that was exactly periodic.
    """
    # Two frames are well defined: the seam and the only inner transition are
    # the same pair, so the ratio is 1.0. Only 0 and 1 frames need rejecting —
    # with a three-frame threshold `--frames 2` printed 0.00, which under the
    # CLI's own label reads as a catastrophic tear rather than a perfect loop.
    if len(frames) < 2:
        return 0.0
    diffs = [
        np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).mean()
        for i in range(1, len(frames))
    ]
    seam = np.abs(frames[0].astype(np.int16) - frames[-1].astype(np.int16)).mean()
    inner = float(np.median(diffs))
    return float(seam / inner) if inner else 0.0
