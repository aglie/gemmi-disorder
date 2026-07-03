# Proposal: windowed (tiled) 3D-PDF evaluation

## Why the current path does not scale

`sf_gemmi` computes a full-box density on a real-space grid, FFTs it, and
keeps the whole reciprocal cube. The reciprocal sampling step is nailed to

    ΔH = 1 / supercell = 1 / N_cells          (r.l.u. of the motif cell)

i.e. Nyquist for a real-space range equal to the **entire box** `L = N_cells·a`.
For a 4 M-atom FCC particle (100³ cells, `L ≈ 361 Å`) at a PDF-grade
`Q_max = 25 Å⁻¹` this is a `3000³` base grid, `5400³` after the FFT pad — a
**2.5 TB** complex buffer. It is the reciprocal *sampling density* that kills
us, and almost all of it is wasted: nobody Fourier-transforms this back to a
PDF that is trustworthy out to 361 Å.

The observation that drives this proposal: **the reciprocal bin size and the
PDF real-space range are a single Fourier pair.** You do not get to pick the
grid from the box; you pick it from the resolution your data actually
supports. Everything below follows from choosing that pair honestly and
never materialising anything bigger than it.

## The two knobs

Two independent numbers describe *any* PDF-oriented calculation. Both come
from the experiment, not from the model:

| knob | set by | controls | value for the target |
|------|--------|----------|----------------------|
| `Q_max` | data resolution | PDF real-space **resolution** `Δr = π/Q_max` | 25 Å⁻¹ → Δr ≈ 0.126 Å |
| `r_max` | PDF range of interest | reciprocal **bin** `ΔQ = π/r_max` | choose per study |

The sampling theorem ties them to the array shapes:

- Reciprocal samples per axis over `[−Q_max, Q_max]`:
  `n = 2·Q_max / ΔQ = 2·Q_max·r_max / π`.
- PDF-window voxels per axis over `[−r_max, r_max]`:
  `2·r_max / Δr` — the same number `n` (they are a DFT pair).

So *one* integer `n` describes both the S(H) map and the PDF map. Sizes for
the target particle (`Q_max = 25`, `a = 3.61 Å`), against the current `3000³`:

| `r_max` | `ΔH` (r.l.u.) | `n` (map/window edge) | S(H) map (cplx) | PDF window (f32) | oversampling saved |
|--------:|-------------:|:---------------------:|:---------------:|:----------------:|:------------------:|
| 20 Å | 0.18 | 319 | 0.5 GB | 0.1 GB | ~735× |
| 30 Å | 0.12 | 478 | 1.7 GB | 0.4 GB | ~218× |
| 50 Å | 0.072 | 796 | 8.1 GB | 2.0 GB | ~47× |
| 100 Å | 0.036 | 1592 | 65 GB | 16 GB | ~6× |

The whole 4 M-atom job collapses from *terabytes* to *gigabytes* the moment
`r_max` is chosen from physics rather than from the box.

### Why `r_max` is small for diffuse work

For a **ΔPDF / diffuse** study the news is even better. The diffuse signal is
`I − |⟨F⟩|²`; its transform is the *difference* between the average Patterson
and the Patterson of the average structure. That difference decays over the
**correlation length** `ξ` of the disorder (nanometres), not over the particle
size. So `r_max ≈ a few·ξ` — the 20–30 Å rows above. Only if you also want the
sharp Bragg/average part rendered as a wide map do you need large `r_max`, and
that part is cheap to obtain from the average structure directly.

## Reframe: compute the PDF window, not the reciprocal cube

Because the two maps are a DFT pair, it is cleaner to compute the **3D-PDF
window directly in real space** and take one small FFT at the end if a
reciprocal `S(H)` map is wanted. The PDF is the density autocorrelation:

    P(r) = ∫ ρ(x) ρ(x + r) dx = (ρ ⋆ ρ)(r)

We only need `|r_α| ≤ r_max`. A correlation whose **output support is tiny
compared to its input support** is the textbook case for *sectioned
(overlap-save) convolution*: tile the input, correlate only the block pairs
that can land inside the output window, discard the rest. "Only keeping track
of the blocks that are needed" is precisely the block-separation cutoff below.

## The correct convolution chain

Everything the FFT path does implicitly must be reproduced, in the right space:

1. **Atomic form factor.** Build `ρ` by placing each atom as its IT92 Gaussian
   sum (exactly what `DensityCalculatorX` already does). In the PDF this
   convolves each pair-vector with the *autocorrelation of the atom shapes*;
   in reciprocal space that is the familiar `|f(Q)|²` weighting. Handled for
   free by working with density rather than point atoms — and it handles mixed
   atom types correctly with no per-pair bookkeeping.

2. **Voxel sampling.** Sampling `ρ` on a `Δr = π/Q_max` grid convolves with the
   voxel box → a `sinc` taper in reciprocal space. At Nyquist for `Q_max` this
   is a sub-percent droop across the band; oversample `Δr` by ~1.2× if it
   matters, or divide it out analytically.

3. **The output window IS the resolution kernel — this is the load-bearing
   convolution.** Truncating `P(r)` to `|r| ≤ r_max` multiplies it by a window
   `w(r)`. In reciprocal space that **convolves** `S(H)` with `W(H) = FT[w]`.
   That convolution is not an artefact to be minimised — it *is* the
   reciprocal-space bin/instrument resolution, and its width is fixed at
   `ΔH ≈ 1/(2 r_max)` by the same Fourier pair. Consequences:
   - use a **smooth** taper (Hann / Gaussian roll-off over the outer shell of
     the window), not a hard box, or `S(H)` rings with `sinc` side-lobes;
   - `r_max` and the S(H) bin cannot be chosen independently — pick `r_max`,
     the bin follows.

4. **blur / deblur.** As today, add a small isotropic `blur` B-factor to each
   atom Gaussian for FFT stability. It convolves `ρ` with a narrow Gaussian, so
   `P = ρ⋆ρ` carries a Gaussian **twice** as wide (autocorrelation of a
   Gaussian). Undo it on the final `S(H)` by `exp(2·blur·s²)` — note the factor
   **2** relative to the amplitude-space `exp(blur·stol²)` in `sf_gemmi`,
   because we deblur an intensity, not an amplitude. (See STYLE_NOTES item 2
   for the amplitude-space form.)

## Algorithm: tiled autocorrelation, near blocks only

Grid pitch `Δr = π/Q_max`; full box is `N_v = L/Δr` voxels/axis (~2873 for the
target — **never stored whole**). Partition into cubic blocks of `B` voxels
(`b = B·Δr`). Let `M = r_max/Δr` be the lag half-window in voxels.

Write `ρ = Σ_i ρ_i` with `ρ_i` the density of block `i`. Then

    P(r) = Σ_{i,j} (ρ_i ⋆ ρ_j)(r)

and `(ρ_i ⋆ ρ_j)` can only reach into `|r| ≤ r_max` when the two blocks are
separated by no more than `b + r_max`. **Every farther pair is skipped** —
that is the whole saving. With `b ≈ r_max` each block only sees its 26
neighbours (`K ≈ 27` pairs, `≈ 14` after using `P(−r)=P(r)`).

```
choose Δr = π/Q_max,  M = ceil(r_max/Δr),  B ≈ M            # block ≈ window
allocate PDF window  Pw  of shape (2M+1)^3           # ~2 GB f32 at r_max=50
for each block i:
    ρ_i  = density of atoms in block i (+ margin M for Gaussian tails)   # gemmi DensityCalculatorX on a sub-box
    for each block j ≥ i within (b + r_max):
        c   = irfftn( rfftn(ρ_i, s) * conj(rfftn(ρ_j, s)) )   # s = (B+2M) padded
        add c into Pw at lag offset (centre_j − centre_i)      # and its mirror if i≠j
apply smooth window w(r) to Pw                        # the resolution kernel (step 3)
S(H) = fftshift(fftn(Pw)) * exp(2·blur·s²) * dV       # optional small reciprocal map
```

Only a handful of `B³` blocks and the `(2M+1)³` output are ever resident.
`ρ_i` is built with the **existing** `sx_to_mx_structure` +
`DensityCalculatorX` machinery, restricted to a sub-box — no new physics, and
the blur/deblur and IT92 handling come along unchanged.

### Cost and memory (target, `r_max = 50 Å`)

Total FFT work is one overlap-save pass: `~ N_v³ · K · log₂(2B) / 2`. With the
measured numpy-FFT constant (`≈ 6.2e-10 s` per `N·log₂N`) this is **~0.5 h per
configuration**, resident memory a few GB — versus **~6 h and ~2.5 TB** for the
monolithic path. Density placement is the same linear-in-atoms cost as today
(≈ built per block), and is overlappable with the FFTs. `r_max = 30 Å` (a
realistic diffuse window) is roughly **8× cheaper** again.

### Alternative: direct pair histogram (DISCUS-style)

For sparse or strongly-cut windows, binning pair vectors `r_j − r_k` (with cell
lists, cutoff `r_max`) straight into `Pw`, then convolving once with the
form-factor autocorrelation, is simpler and avoids the block FFTs. Cost
`~ N · ρ·(4/3)π r_max³` pairs (≈ `1.8e11` at `r_max = 50`, linear in `N`,
trivially parallel). The density-FFT route above is preferred as the primary
plan because it reuses gemmi's exact multi-element form-factor handling; the
histogram is the fallback when `r_max` is small enough that pair counts win.

## Diffuse / ΔPDF specifics

Per configuration `c` accumulate, on the shared `r`-window:

    ⟨P⟩ += P[ρ_c] / N          (average Patterson)
    ⟨ρ⟩ += ρ_c   / N           (running average density, block-wise)

At the end `ΔPDF = ⟨P⟩ − P[⟨ρ⟩]`, which equals `FT[⟨I⟩ − |⟨F⟩|²]` on the
window — the same three maps `average_diffuse` produces today, but only where
the signal lives. Because `ΔPDF` is short-ranged, `r_max` here is set by `ξ`,
so this is the cheapest mode of all.

## How it lands in the package

- `Grid.from_pdf(motif_cell, q_max, r_max)` — the honest constructor: derives
  `Δr`, `ΔH`, `n`, and the block size, replacing the box-driven
  `from_supercell` for PDF work.
- new `gemmi_disorder/pdf.py`: `tiled_patterson(structure, pdfgrid, blur, window)`
  and a `pdf_diffuse(structures, pdfgrid, …)` mirroring `average_diffuse`.
- `save_to_yell` gains an `is_direct=True` branch to write the real-space PDF
  window (the format already carries `lower_limits`/`step_sizes`, so this is a
  flag, not a new schema).

## Validation

1. On a small case where the monolithic path *fits* (e.g. 20³ cells,
   `r_max = box`), the tiled `S(H)` must equal `sf_gemmi`'s `|F|²` to numerical
   precision — this checks the convolution chain and the deblur factor-2.
2. Shrinking `r_max` must reproduce the *windowed* transform of that same
   reference (reference `P(r)` multiplied by `w`), confirming step 3.
3. `sf_gemmi_direct` on a handful of `H` inside the band remains the ground
   truth for absolute scale (`F(000) = Σ Z`, STYLE_NOTES item 1).

## Open choices

- Window shape (Hann vs Gaussian vs Lanczos) and how much of the outer shell to
  taper — trades PDF ripple against effective `r_max`.
- Whether to oversample `Δr` slightly (~1.2×) to make the voxel `sinc` and any
  block-edge effects negligible, at a modest memory cost.
- Block size `B` vs neighbour count `K`: the overlap-save optimum (larger
  blocks, fewer neighbours) is machine-dependent; expose it as a tunable.
- Anisotropic windows: disorder often has different `ξ` along different axes, so
  allow per-axis `r_max` (the code already allows per-axis `hkl_max`).
