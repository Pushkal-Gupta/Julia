# Noise and preprocessing

A question I'd been carrying around since I started Canny: how much
does the choice of pre-smoother *actually* matter? The Canny pipeline
has a Gaussian built in. Is there any point adding a second smoother
on top, and if so, which one?

`examples/08_noise_lab.jl` answers this directly. Same clean image,
same Canny parameters, same tolerance for edge matching. The only
thing that varies is the noise type and the smoother.

## Setup

Ground truth: Canny on the clean image, with my standard parameters
(σ=1.0, low=0.06, high=0.18). On the test image (a disk plus a
square) that gives a 319-pixel edge map.

Then I add noise — Gaussian (σ=0.12) or salt-and-pepper (p=0.08) —
and run the same Canny on the result, optionally with one of eight
smoothers applied first. Each output gets scored against the ground
truth via precision / recall / F1 at a 1-pixel tolerance.

## What happened with Gaussian noise

| Smoother      | edge pixels | precision | recall | F1    |
|---------------|-------------|-----------|--------|-------|
| none          | 299         | 0.910     | 0.994  | 0.950 |
| box 3×3       | 271         | 0.989     | 0.991  | 0.990 |
| box 5×5       | 280         | 0.989     | 0.984  | 0.987 |
| Gaussian σ=1  | 273         | 0.985     | 0.991  | 0.988 |
| Gaussian σ=2  | 285         | 0.972     | 0.984  | 0.978 |
| median 3×3    | 267         | 0.996     | 0.984  | 0.990 |
| median 5×5    | 267         | 0.993     | 0.978  | 0.985 |
| bilateral     | 273         | 0.996     | 0.994  | **0.995** |

Three observations:

1. Even without extra smoothing, Canny does OK (F1 = 0.95). Its
   internal Gaussian at σ=1.0 already handles mild Gaussian noise.
2. *Any* of the smoothers pushes F1 above 0.98. The differences
   between them are small — fractions of a percent.
3. Bilateral wins, just barely. The intensity-aware weighting lets
   it smooth flat regions hard while not blurring real edges, which
   shows up as slightly better precision *and* recall.

If I had a budget of one pre-smoother for Gaussian noise, bilateral
is the winner. But the verdict is "you probably don't need one".

## What happened with salt-and-pepper noise

| Smoother      | edge pixels | precision | recall | F1    |
|---------------|-------------|-----------|--------|-------|
| none          | 2329        | 0.119     | 0.987  | 0.212 |
| box 3×3       | 1207        | 0.225     | 0.987  | 0.366 |
| box 5×5       | 457         | 0.600     | 0.987  | 0.746 |
| Gaussian σ=1  | 862         | 0.313     | 0.987  | 0.476 |
| Gaussian σ=2  | 359         | 0.772     | 0.987  | 0.866 |
| median 3×3    | 336         | 0.955     | 0.997  | 0.976 |
| median 5×5    | 314         | 0.981     | 0.997  | **0.989** |
| bilateral     | 2336        | 0.119     | 0.987  | 0.212 |

This is the more interesting half of the experiment.

**Without a smoother, Canny falls apart.** 2329 edge pixels vs 319 in
the ground truth — every speck of salt and pepper becomes a tiny
edge cluster. Precision drops to 12%.

**Box and Gaussian help, but they're playing the wrong game.** They
spread each speck into a blob, which trades sharp false-positive
edges for blurred false-positive edges. Bigger box / bigger σ helps
but you pay in localization.

**Median crushes salt-and-pepper.** F1 jumps from 0.212 (no
smoother) to 0.989 (median 5×5). This is exactly what median is for:
a single outlier pixel in a 3×3 window can't pull the output past
the median, so individual specks vanish.

**Bilateral fails — and the failure is instructive.** F1 = 0.212,
identical to "no smoother". Bilateral preserves anything that looks
like an intensity edge, and a salt speck has the strongest possible
intensity edge against its neighbors. So bilateral *preserves the
noise*. The same property that makes it the winner on Gaussian noise
makes it useless on impulse noise.

## What I'd actually use

- Gaussian noise: don't bother adding a second smoother — Canny's
  internal Gaussian is enough. If I really want the extra 4 points
  of F1, bilateral.
- Salt-and-pepper noise: median, every time. 5×5 if the noise is
  bad, 3×3 if it's light.
- Mixed noise (real-world): median first, then optionally bilateral
  for the Gaussian component. The order matters — median doesn't
  spread the salt out the way Gaussian does, so the bilateral sees a
  cleaner image after.

## The metric matters

I'm using F1 at 1-pixel tolerance, which is generous — it doesn't
care which side of an edge you landed on, only that you landed
within one pixel. If I tightened tolerance to 0:

- Small differences in smoothing scale would show up as recall
  drops (real edges shifted by a pixel become misses).
- Box vs bilateral vs Gaussian distinctions would matter more.
- The story would change less for salt-and-pepper (median still
  wins) and more for Gaussian (bilateral's edge-preservation would
  pull ahead further).

The 1-pixel tolerance is the right call for "did the smoother help me
find the edges". A 0-pixel tolerance would ask "did the smoother
help me find the edges *in the exact same place*", which is a
different (and stricter) question.

## What this experiment doesn't cover

A few caveats so I don't over-generalize from one image:

- Real-world images have textured regions, which median treats as
  noise and over-smooths. The synthetic test image has no texture, so
  median looks better here than it would in practice.
- I held Canny parameters fixed. In a real pipeline you'd retune the
  thresholds after switching smoothers. Some of the differences
  between similar smoothers (box 3×3 vs Gaussian σ=1) would shrink.
- I only tested two noise types at one intensity each. The story for
  speckle, Poisson, or mixed noise is different.

But the qualitative ordering (median crushes impulse; bilateral wins
on Gaussian; box / Gaussian sit in between) is robust across the
parameter changes I tried.
