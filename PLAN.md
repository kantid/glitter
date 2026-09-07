# glitter — design notes

Turn any picture into the tackiest possible sparkling card from 2008, and make
it something anyone can reproduce with one command.

This document records the decisions, the measurements behind them, and the
places where the original plan turned out to be wrong.

---

## Three decisions that shape the whole architecture

These are not optimisations for later. Skip them at the start and the thing has
to be rewritten.

### 1. Glitter does not move — it twinkles

Real glitter is glued to the surface. Only its brightness changes. Which means
the particles **do not need to be redrawn every frame**.

```
particles (32,000)
        |
        +-- scattered across K = 16 phase buckets
        |
        v
16 static layers, rendered ONCE   (uint8 RGB, ~4.7 MB each)
        |
        v
frame(t) = base + sum_k w_k(t) * layer_k,  w_k(t) = 0.5 + 0.5*sin(phi_k + w_k*t)
```

Sixteen array multiplications per frame instead of 32,000 draw calls.
**~0.15s per frame instead of ~10s.** The whole loop renders in ten seconds
rather than ten minutes.

Only rotating stars fall outside the scheme; they get 8 discrete angles in the
sprite atlas and twinkle instead of spinning. Sweeps and the flash are
full-frame gradients, cheap in their own right.

### 2. The loop must be seamless — which constrains every frequency

Fix a cycle length of `N` frames. **Only** frequencies that divide the cycle
are allowed:

```
w_m = 2*pi*m/N,  m in 1..6
```

A sweep makes a whole number of passes per cycle. The flash fires exactly once
per cycle. Arbitrary speeds are forbidden at the API level: the particle
generator takes `m`, not `speed`.

Default is `N = 48` frames at 20 fps, so 2.4 seconds.

### 3. GIF quantisation is not an export step, it is half the look

A pink gradient plus point highlights in 256 colours gives banding and
dithering mush.

- Global palette sampled from **all** frames, not the first.
- `dither=NONE`. Dithering smears sparks into dirt; 128 colours without it beat
  256 with it.
- Delta encoding through a transparent index plus `disposal=1`.
- Downscale the GIF. Resolution and frame count are the real size levers.

### What measurement showed (and where this plan was lying)

A run on the reference card, 1120x1405, 48 frames, 38k particles:

| | promised | actual |
|---|---|---|
| frame render | ~150 ms | **58 ms** |
| GIF 640px / 128 colours | 4–6 MB | **21.7 MB** |
| WebP at full size | 1–1.5 MB | **29.4 MB** |

The same reasoning error underlay both: glitter is maximum-entropy noise and
nothing compresses it. Two corrections followed.

**Pillow's `optimize=True` is useless, but delta encoding is not.** Measurement
showed that **72% of pixels match between adjacent frames** — wherever there is
no glitter, the frame is literally the base image. Pillow's `optimize` cannot
exploit that, since all it does is crop a frame to the bounding box of the
changes, and glitter is smeared across the whole area. Marking the matching
pixels with a transparent index by hand gives **−51%**, completely losslessly.

**WebP is not four times smaller than GIF. It is not smaller at all.** Its
lossy compression does poorly on high-frequency noise, and its inter-frame
prediction cannot predict random sparks. At 640px: GIF with delta 10.6 MB
against WebP q80 at 12.4 MB. WebP wins only on colour fidelity, having no
64-colour palette to squeeze into.

The real size table, 48 frames:

| | 480px | 640px |
|---|---|---|
| GIF 64 colours, no delta | 10.6 MB | 17.7 MB |
| GIF 64 colours, with delta | **5.0 MB** | **8.3 MB** |

Resulting defaults: GIF at 560px / 64 colours / delta, WebP at 900px / q70.

Caveat: the source used for measurement is itself a glitter card, so it is the
worst case for entropy. On an ordinary photograph the numbers come out
noticeably lower.

---

## Layout

```
glitter/
  cli.py         # arguments, presets, --preview
  analyze.py     # density maps, numpy/scipy
  sprites.py     # atlas: dots, specks, 4/6/8-ray stars
  layers.py      # phase buckets, sweeps, flash, colour grade
  render.py      # frame assembly
  export.py      # quantisation, gif / webp
  presets/*.toml
```

`particles` / `stars` / `sweep` are **not separate modules**. They are 80% the
same code (sprite + position + phase), and what separates them is lines in TOML.

---

## Masks: an honest account of what is possible

Without a neural network, a picture yields statistics rather than semantics. So
the density map is a weighted sum, not a `flower_mask`:

```
density = w1*luma^gamma
        + w2*saturation
        + w3*edge_energy        (Sobel + blur)
        + w4*hue_band(magenta +/- 25 degrees)
        - w5*protect
```

What actually works:

| goal | how |
|---|---|
| roses sparkle harder | `hue_band` around magenta, times high saturation |
| gold glints | `hue_band` around yellow, times high luma |
| white lettering shines | morphological top-hat: bright, thin strokes, high contrast |
| background sparkles less | percentile normalisation clips the bottom of the range |
| faces left alone | `--protect mask.png` **or** the HSV skin heuristic |

Face detection without a neural network is not possible. So it becomes a
feature instead: a manual exclusion mask, with the skin heuristic as the
default. For other people's pictures that is arguably more useful anyway.

---

## Stages

The riskiest parts — performance and export — come **first**, not last. Each
stage improves an existing layer rather than bolting on a new one.

**Stage 0 — end-to-end slice.** Image, uniform mask, 200 dot particles, phase
buckets, 48 frames, a GIF on disk. Ugly, but working end to end, with the
architecture from points 1 and 2 already in place. Done.

**Stage 1 — masks.** `analyze.py`: luma, saturation, Sobel, hue bands, top-hat,
skin heuristic, `--protect`. Weighted position sampling by density. Done.

> Surfaced along the way: at `text = 0.40` the top-hat lands squarely on the
> letter strokes and the greeting stops being readable. A card with unreadable
> text is a failure, so the weight dropped to 0.22 and `density_gamma` was
> pulled forward from stage 3 — the exponent each system raises the shared map
> to before sampling. Large particles get `0.6`, spreading them out and off the
> thin strokes; fine dust gets `1.3` and crowds into bright places instead. One
> line per system, no recomputation of the map.

**Stage 2 — sprites.** Atlas: gaussian-cored dot, 4/6/8-ray stars with a bloom.
Eight angles per size, pre-rendered once, indexed thereafter. Done.

> A ray is computed from the distance to its axis rather than from an angular
> lobe of the form `|cos(n*theta/2)|^p`: a lobe has constant angular width,
> which means in pixels it widens linearly with radius and ends as a wedge
> rather than a needle.
>
> Two settings had to be found from a contact sheet of the sprites: a falloff
> exponent of 1.5 along the ray dropped the middle to 35% brightness and the
> star read as a blob (now 0.9); a bloom with sigma at 0.22 of the size
> swallowed the short rays of `star8`, making the eight-ray star
> indistinguishable from the four-ray one (now 0.10).
>
> And a bug that only appeared with large stars: `floor` was applied AFTER
> protection, so it cancelled protection out — a protected area received the
> floor rather than zero. At stage 1, with small dots, this was invisible; with
> big stars a star grew in the subject's beard. The floor is now raised before
> protection.

**Stage 3 — particle systems.** fine / medium / stars / big stars as TOML
entries over one shared sampler. `density_gamma` already exists, pulled forward
at stage 1.

**Stage 4 — sweeps.** Diagonal bands of light crossing the frame. Multiplicative
brightness lift, a whole number of passes per cycle. Done.

> A sweep's brightness depends only on the pixel's projection onto one axis,
> that is on a single scalar. So there is no need to evaluate full-frame
> trigonometry: each frame fills a 4096-entry table and the frame is an index
> lookup through a precomputed map. Three sweeps cost +11 ms per frame instead
> of doubling it.
>
> Glitter under a sweep is lifted by a separate, larger coefficient than the
> background (1.4 against 0.3) — otherwise this is not a beam of light but a
> plain exposure bump.
>
> Keep `width` narrow. A gaussian profile has long tails, and on wide bands
> they add up into a constant wash across the whole area: the field median
> drifts to 0.34, glitter is lifted by half again everywhere, and the card
> washes out. Aim for a field median near zero with a peak of 1.5 to 2.

**Stage 4.5 — soft highlight knee.** Not in the original plan; it surfaced while
investigating the washing out. The sweeps turned out to be innocent:
measurement showed **14% of the frame was already pure white without them**.
Additive glitter overshoots the ceiling, and hard clipping there removes colour
rather than brightness — on a pink pixel the red channel saturates first and
the rose turns white. A knee (identity to 205, asymptotic to 255 above) removes
this entirely: pure-white pixels drop to 0.0%. Computed with the same kind of
table as the sweeps, since `exp` over 4.7M elements costs more than indexing a
1024-entry array. Done.

**Stage 5 — life.** Saturation / contrast / hue-shift pulsing. Flash. Gentle by
default (1.35x brightness, not a white screen) — a full-frame strobe for 100 ms
both reads as a bug and is dangerous for photosensitive viewers. The extreme
lives behind `--flash-extreme` with a warning in the README. Done.

> Saturation, contrast, hue shift and the flash are all linear operations, so
> they fold into **a single 3x3 matrix with an offset** and are applied with one
> matmul instead of four passes over the frame. The whole stage costs +12 ms
> per frame.
>
> The three phases are spread a third of a period apart: in lockstep they read
> as one shared brightness pulse and having three separate knobs stops meaning
> anything. Hue rotation is about the grey axis, so grey stays grey.
>
> A trap: `np.matmul` with an `out` that overlaps the input gives an undefined
> result. It needs a separate buffer.
>
> Verified numerically rather than by eye: hue rotation preserves grey at every
> angle, the flash fires exactly once per cycle and is periodic, the grade at
> frame 0 matches frame N bit for bit, and an identity grade is skipped at no
> cost.

**Stage 6 — export.** Global palette, GIF and animated WebP, honest sizes and
`--width`. Done.

**Stage 7 — project.** Presets, `--seed`, `--list-presets`, a README with
before/after, and a legal note about other people's photographs. Done.

> Of the presets, only `default` is documented in full — it is also the
> reference that others are copied from; the rest point at it briefly so the
> descriptions cannot drift apart.
>
> The first `maximum-overdrive` was not "maximum glitter" but a destroyed
> picture: 38.8% of the frame near-white, saturation 0.23 against 0.54 in the
> source. Retuned by the numbers rather than by eye, it came out at 24.5% and
> 0.31. The progression by share of near-white pixels: tasteful 3.8%, default
> 14.8%, overdrive 24.5%.

---

## Testing on unfamiliar pictures

Running two outside cards (pumpkins with a cat, dogs with cucumbers) through
the pipeline showed that the "share of near-white pixels" metric is blind to
the real problem. On the pastel dog card it was *lower* than on the reference
card (22.7% against 24.5%), yet the picture looked destroyed.

The metric that works is the **correlation between the result's luminance and
the source's**, that is, how much structure survived:

| | default | maximum-overdrive |
|---|---|---|
| reference card | 0.81 | 0.72 |
| pumpkin card | 0.69 | 0.57 |
| pastel card | 0.66 | 0.55 |

The reference card is high-frequency and saturated to begin with, so glitter
has nothing to drown out. On smooth pale areas it becomes the only detail and
erases the picture.

Hence `headroom`: a speck's gain is damped by the brightness of where it
landed. White cannot be made whiter. It improves all six combinations and most
where things were worst — the pastel card at overdrive goes 0.55 to 0.62, at
default 0.66 to 0.73.

Separately: `--preview` took exactly 25% of the resolution, which on a 700px
picture gave 175px — lying about precisely the thing it exists for. There is
now a 420px floor.

## Diamonds and glare

Two complaints about the output, both real defects, with different causes.

**Diamonds.** A three-pixel dot cannot look like anything else: centre 1.0,
four sides 0.54, four corners 0.29 — a circle cannot be drawn on nine pixels.
On top of that a gaussian with sigma at 0.30 of the size was cut off at 25%
brightness at the rim, and that hard edge is what read as a polygon.

Fixed by supersampling: the sprite is evaluated on a grid four times finer and
averaged down. Sigma dropped to 0.22 as well, leaving 8% at the rim instead of
25%. It costs nothing — the atlas is built once.

A hypothesis that Lanczos was to blame when downscaling the GIF (it has
negative lobes along the axes and could have produced cross-shaped ringing) was
tested and rejected: LANCZOS, BICUBIC and BOX gave indistinguishable results.

**Glare.** The first instinct was to lower the fine dust's gain, and it changed
almost nothing: 10.2% of pixels brighter than 245, against 9.4%. Ablating the
systems one at a time found the actual culprit:

| source | contribution to clipping |
|---|---|
| sweeps | **+4.2** |
| medium dots | +3.7 |
| fine dust | +3.4 |
| all stars together | +1.7 |

Sweeps multiplied the glitter beneath them by 2.4. Dropping that to 1.05 took
clipping from 10.2% to 8.8%, and surviving structure rose from 0.81 to 0.85
along the way.

## Checking against the original technique

Antialiasing removed the diamonds, but the shape still read as foreign.
Studying the original technique (Paint Shop Pro and Jasc Animation Shop,
mid-2000s tutorials) explained why — everything there worked differently:

| then | what we had |
|---|---|
| region flood-filled with a seamless noise tile | a scatter of individual sprites |
| three frames at 5–9 fps, whole field swapped at once | 48 frames, a sinusoid per particle |
| indexed palette, a pixel is lit or not | gaussian midtones |
| **a hard bright border around the filled region** | no border at all |

The last row turned out to be the important one: glitter-text tutorials repeat
`Selections > Expand` by one pixel and "it looks best when you have a border"
verbatim. What sparkles is a region with an edge, not the dots themselves.

Hence two additions: the `speck` sprite (a hard pixel, antialiasing
deliberately off) and the `[contour]` section — an outline around regions above
a density threshold, scattered across buckets pixel by pixel so it shimmers
rather than pulsing as one piece. Placed vectorised: there are close to thirty
thousand outline pixels, and a per-pixel `_stamp` would eat the entire saving
the buckets bought.

The `y2k` preset brings this together.

## The seam metric was lying

The new preset reported a seam of 1.35 on a loop that is periodic exactly. The
metric was at fault: it compared the seam against a **single** 0 to 1
transition. With three phase slots and high harmonics the transitions vary a
great deal, and a coincidentally small first one inflated the ratio. The
baseline is now the median across all transitions, after which every preset
lands between 0.76 and 1.11.

## What the code review found

Six findings, all reproduced before being fixed.

**The outline drew a frame around the whole picture.** scipy's morphology
treats everything past the image border as background, so a region reaching the
edge was eroded from outside and `dilation & ~erosion` lit an outline where no
border exists. On a uniform density with no border at all this produced 448
outline pixels; on the pastel card it lit a quarter of every edge pixel. Fixed
by padding the image with the border replicated outward before the morphology:
448 to 0, and 25% to 3% on the card (the remainder being genuine contours that
do reach the edge).

**The outline geometry did not scale with resolution.** The structuring
elements were a fixed 5x5 and 3x3 and `width` was in absolute pixels, while
particles are counted per megapixel and the text stroke scales with the frame
side. In other words `--preview` was lying in exactly the way already fixed
once before. Sizes are now specified for an 800px side and scaled from there.

**`width` was twice what the comment promised:** `dilation(w) & ~erosion(w)`
gives w pixels outward plus w inward. It is a radius, not a thickness; the
comments were corrected and the behaviour left alone.

**Empty `contour.harmonics` with `strength > 0`** crashed `rng.choice` with a
bare traceback, while every other config error exits with a readable message.

**The seam metric returned 0.00 for two frames** — which, under the CLI's own
label, reads as a catastrophic tear, although for two frames it is well defined
and equals 1.0. The cutoff was one frame too high.

**`y2k` pointed at documentation that did not exist:** the `[contour]` section
was described nowhere but in `y2k` itself. It was added to `default.toml`,
disabled.

## A side observation

Source saturation is 0.54, the result's 0.39. This is not a bug: glitter is
white and inevitably lightens whatever it lands on. Recovering it would be a
preset knob — less white in the systems' palettes.

## Risks

- **New Python releases**: scipy and numpy wheels sometimes lag on the very
  latest. If scipy will not install, `gaussian_filter` and the morphology are
  replaceable with separable convolution and dilation in pure numpy.
- **RAM**: 16 buckets at 1120x1405x3 uint8 is 75 MB. Acceptable. In float32 it
  would be 300 MB, which is why the layers are stored as uint8 and the weights
  applied on the fly.
