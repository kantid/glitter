"""Configuration and presets.

Every animation parameter lives in TOML rather than in code: a preset is a set
of particle systems layered on top of one shared sampler (see PLAN.md).
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

PRESETS_DIR = Path(__file__).parent / "presets"


@dataclass
class SystemSpec:
    """One particle system. All that separates fine/medium/stars lives here."""

    name: str
    #: Particles per megapixel, not an absolute count — see layers.particle_count.
    count: int
    sprite: str = "dot"
    size_range: tuple[int, int] = (3, 7)
    gain: float = 1.0
    #: Exponent this system raises the shared density map to before sampling.
    #: Above 1 the system crowds into the hottest spots, below 1 it spreads out
    #: more evenly. That is how large particles are pulled off the thin strokes
    #: of lettering without touching the fine dust.
    density_gamma: float = 1.0
    #: Harmonic numbers this system may use. Integers only, or the loop tears.
    harmonics: tuple[int, ...] = (1, 2, 3)
    colors: tuple[str, ...] = ("#ffffff",)


@dataclass
class SweepSpec:
    """A band of light travelling across the card."""

    angle: float = 35.0  # degrees; 0 is a vertical band moving right
    bands: float = 3.0  # how many bands fit on screen at once
    #: Passes per cycle. INTEGERS ONLY, otherwise the sweep jumps at the seam.
    passes: int = 1
    width: float = 0.10  # fraction of a period; sigma of the gaussian profile
    strength: float = 1.0


@dataclass
class ContourSpec:
    """A shimmering outline around the glittered regions.

    Outline pixels are scattered across phase buckets individually, so the
    outline shimmers rather than pulsing as one piece — like the hard border
    around a glitter-filled selection in the cards of that era.
    """

    strength: float = 0.0
    threshold: float = 0.55  # cutoff on the density map
    #: Outline RADIUS: it extends `width` pixels outward from the region border
    #: and the same distance inward, so the ring is twice that thick. Specified
    #: for an 800px frame side and scaled from there.
    width: int = 1
    harmonics: tuple[int, ...] = (1, 2)
    colors: tuple[str, ...] = ("#ffffff",)


@dataclass
class GradeSpec:
    """Colour pulsing. These are amplitudes, not absolute values."""

    saturation: float = 0.0
    contrast: float = 0.0
    hue: float = 0.0  # degrees
    #: Harmonic of the pulse. Integer, or the loop tears.
    harmonic: int = 1


@dataclass
class FlashSpec:
    """One flash per cycle.

    Gentle by default: a brightness lift, no white. A full-frame white strobe
    both reads as a bug and is dangerous for photosensitive viewers, so it
    lives behind `white` and the --flash-extreme flag.
    """

    strength: float = 0.0  # peak brightness lift: 0.4 means x1.4
    frames: float = 3.0  # duration in frames
    at: float = 0.0  # position in the cycle, 0 to 1
    white: float = 0.0  # white added at the peak, as a fraction of 255


@dataclass
class HueBand:
    """A hue band: the closer a pixel sits to `center`, the more glitter it gets."""

    center: float  # degrees; 0 is red, 60 yellow, 300 magenta
    width: float  # degrees; sigma of the gaussian
    weight: float


@dataclass
class DensitySpec:
    """Feature weights for the density map. Tune them with --dump-density."""

    luma: float = 0.35
    luma_gamma: float = 2.0
    saturation: float = 0.25
    edges: float = 0.20
    text: float = 0.35
    #: How much to suppress glitter on skin tones. 1.0 leaves them untouched.
    protect_skin: float = 0.9
    #: Density floor: the background should still sparkle, just far less.
    floor: float = 0.03
    hue_bands: list[HueBand] = field(default_factory=list)


@dataclass
class Config:
    #: Cycle length in frames. Fixes every legal frequency: w_m = 2*pi*m/frames.
    frames: int = 48
    fps: int = 20
    #: Phase bucket count = len(harmonics) * phase_slots (see layers.py).
    phase_slots: int = 4
    seed: int = 0
    #: How much a sweep lifts the picture itself and, separately, the glitter
    #: under it. Glitter has to flare noticeably harder than the background,
    #: otherwise this is not a beam of light but a plain exposure bump.
    sweep_base_gain: float = 0.30
    sweep_glitter_gain: float = 1.40
    #: Knee threshold for highlight rolloff. 255 means hard clipping.
    highlight_knee: float = 205.0
    #: How much to dim a speck by the brightness of where it landed. White
    #: cannot be made whiter: without this, glitter on smooth bright areas is
    #: the only detail left and it erases the picture's structure. 0 disables.
    headroom: float = 0.7
    density: DensitySpec = field(default_factory=DensitySpec)
    contour: ContourSpec = field(default_factory=ContourSpec)
    grade: GradeSpec = field(default_factory=GradeSpec)
    flash: FlashSpec = field(default_factory=FlashSpec)
    systems: list[SystemSpec] = field(default_factory=list)
    sweeps: list[SweepSpec] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.frames / self.fps


def available() -> list[str]:
    return sorted(p.stem for p in PRESETS_DIR.glob("*.toml"))


def load(name_or_path: str) -> Config:
    path = Path(name_or_path)
    if not path.exists():
        path = PRESETS_DIR / f"{name_or_path}.toml"
    if not path.exists():
        raise SystemExit(
            f"preset not found: {name_or_path} (available: {', '.join(available())})"
        )

    raw = tomllib.loads(path.read_text())
    systems = [
        SystemSpec(
            name=name,
            count=s["count"],
            sprite=s.get("sprite", "dot"),
            size_range=tuple(s.get("size_range", (3, 7))),
            gain=s.get("gain", 1.0),
            density_gamma=s.get("density_gamma", 1.0),
            harmonics=tuple(s.get("harmonics", (1, 2, 3))),
            colors=tuple(s.get("colors", ("#ffffff",))),
        )
        for name, s in raw.get("system", {}).items()
    ]
    d = dict(raw.get("density", {}))
    bands = [HueBand(**b) for b in d.pop("hue_band", [])]
    density = DensitySpec(**d, hue_bands=bands)

    sweeps = [SweepSpec(**s) for s in raw.get("sweep", [])]
    for s in sweeps:
        if s.passes != int(s.passes):
            raise SystemExit(f"sweep.passes must be an integer, got {s.passes}")

    contour = ContourSpec(**{
        k: (tuple(v) if isinstance(v, list) else v)
        for k, v in raw.get("contour", {}).items()
    })
    if contour.strength > 0 and not contour.harmonics:
        raise SystemExit("contour.harmonics cannot be empty when strength > 0")

    return Config(
        frames=raw.get("frames", 48),
        fps=raw.get("fps", 20),
        phase_slots=raw.get("phase_slots", 4),
        seed=raw.get("seed", 0),
        sweep_base_gain=raw.get("sweep_base_gain", 0.30),
        sweep_glitter_gain=raw.get("sweep_glitter_gain", 1.40),
        highlight_knee=raw.get("highlight_knee", 205.0),
        headroom=raw.get("headroom", 0.7),
        density=density,
        contour=contour,
        grade=GradeSpec(**raw.get("grade", {})),
        flash=FlashSpec(**raw.get("flash", {})),
        systems=systems,
        sweeps=sweeps,
    )


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
