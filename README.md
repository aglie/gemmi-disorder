# gemmi-disorder

A small wrapper around [gemmi](https://github.com/project-gemmi/gemmi) for
computing **diffuse scattering** from collections of disordered atomic
configurations. The calculation is done by averaging structure factors and
intensities over many realisations of a disordered structure to obtain:

- ⟨F⟩ — average complex structure factor (the Bragg peaks).
- ⟨I⟩ — total scattering.
- ⟨I⟩ − |⟨F⟩|² — the diffuse scattering map.

Built on gemmi's density-calculator fast FFT-based algorithm (the "fast path"),
with X-ray, **neutron**, or electron scattering selectable per call (see
[Choosing the radiation](#choosing-the-radiation-x-ray--neutron--electron)). A
slow but exact `sf_gemmi_direct` which calculates structure factors by direct
summation is also included for validation.

## Install

```bash
pip install -e .
# optional ASE bridge for MC / ORB workflows:
pip install -e .[ase]
```

## Usage

### From CIF files

**One CIF = one configuration.** `gemmi.read_small_structure` reads a single
crystal structure; the averaging loop in this package treats every file in
the list as one realisation of the disorder. Multi-model / multi-block
files are not supported in v1.

Each CIF must:
- be in P1 (no symmetry is applied),
- contain the **full supercell** of atoms (every site explicit, no
  symmetry mates implied), and
- have a unit cell that matches the chosen `supercell` and the underlying
  motif. If the motif is `(a, b, c, α, β, γ)` and `supercell = (nx, ny, nz)`,
  the CIF cell must be `(a·nx, b·ny, c·nz, α, β, γ)`.

The grid is built around the supercell convention (1 r.l.u. step = 1
supercell vector), so the chosen `hkl_max` is in units of the underlying
motif's reciprocal lattice.

```python
import gemmi, glob
from gemmi_disorder import Grid, average_diffuse, save_to_yell

# Each CIF = one snapshot; all snapshots must share the same supercell.
cif_paths = sorted(glob.glob("structures/*.cif"))
structures = [gemmi.read_small_structure(p) for p in cif_paths]

supercell = (6, 6, 12)

# hkl_max can be a single int (same range on all three axes) or a triple
# (hmax, kmax, lmax) to set per-axis ranges — useful when the disorder
# extends further along some directions than others.
grid = Grid.from_supercell(supercell=supercell, hkl_max=14)
# e.g. grid = Grid.from_supercell(supercell=supercell, hkl_max=(14, 14, 8))

result = average_diffuse(structures, grid, blur=0.01, b_iso=0.0, progress=True)

# result holds three averaged maps:
#   result.diffuse      — ⟨I⟩ − |⟨F⟩|²   (the diffuse map, what you usually want)
#   result.average_bragg — |⟨F⟩|²        (the average / Bragg pattern)
#   result.average_I     — ⟨I⟩           (total scattering)
# save_to_yell writes all three to Yell-1.0 .h5 files in <prefix>/:
#   diffuse_intensity.h5   av_intensity.h5   tot_intensity.h5
save_to_yell(result,
             cell=structures[0].cell,
             supercell=supercell,
             prefix="output")
```

### From scatterer coordinates

```python
from gemmi_disorder import Grid, DisorderedStructure, average_diffuse

structures = [
    DisorderedStructure(
        cell_parameters=(4.07, 4.07, 4.07, 90, 90, 90),
        atoms=[("Au", 0.0, 0.0, 0.0), ("Cu", 0.0, 0.0, 1.0), ...],
        supercell=(20, 20, 20),
    )
    for snapshot in mc_history
]
grid = Grid.from_supercell(supercell=(20, 20, 20), hkl_max=2)
result = average_diffuse(structures, grid, blur=0.01)
```

### Choosing the radiation (X-ray / neutron / electron)

By default the atoms scatter as **X-rays** (IT92 form factors). To compute
**neutron** scattering instead — atoms weighted by their bound coherent
scattering lengths (gemmi's `Neutron92`) — pass `scattering="neutron"`. Nothing
else about the call changes; only the per-atom weights differ, and blur/deblur
and scaling are identical.

```python
# Neutron diffuse scattering (X-ray is the default):
result = average_diffuse(structures, grid, blur=0.01, scattering="neutron")

# The switch is also available on the low-level single-config call:
from gemmi_disorder import sf_gemmi
sf = sf_gemmi(structures[0], grid, blur=0.01, scattering="neutron")
```

Accepted values are `"xray"` (default), `"neutron"`, and `"electron"`; an
unknown name raises `ValueError`. The same `scattering=` argument is accepted by
`tiled_patterson` below.

## Large objects: the tiled 3D-PDF path

The `sf_gemmi` fast path samples reciprocal space at `1 / supercell` — Nyquist
for the *whole box*. For a big object (e.g. a 4 M-atom nanoparticle) at
PDF-grade `Q_max` that is a multi-terabyte FFT, and it massively oversamples:
a PDF study only needs reciprocal sampling fine enough for the real-space range
`r_max` you actually trust. The reciprocal bin and `r_max` are a Fourier pair,
so the calculation should be driven by that pair, not by the box.

`tiled_patterson` computes the 3D-PDF (Patterson) window
`P(r) = Σ_x ρ(x) ρ(x + r)` directly, for `|r_α| ≤ r_max`, via an overlap-save
decomposition — never materialising anything bigger than the output window.
See [`docs/tiled_3dpdf_proposal.md`](docs/tiled_3dpdf_proposal.md) for the
derivation and cost model.

```python
import gemmi
from gemmi_disorder import tiled_patterson

structure = gemmi.read_small_structure("nanoparticle.cif")   # P1, orthogonal
pw = tiled_patterson(
    structure,
    q_max=25.0,          # data resolution (Å⁻¹) -> PDF voxel Δr = π/q_max
    r_max=50.0,          # PDF range of interest (Å) -> sets the reciprocal bin
    blur=0.5,            # extra B for FFT stability (left in — see below)
    disk_budget_bytes=8 * 1024**3,   # cap the on-disk block cache; evicted
    mem_blocks=4,                    #   blocks are rebuilt from atoms as needed
)
# pw.data is the raw (2·M+1)³ Patterson window; pw.r_step, pw.r_max, pw.blur.
```

**Deferred corrections.** `tiled_patterson` returns the *raw, blurred*
Patterson: **no deblur and no window taper are applied in 3D**. Both are
point-wise and are best done at the very end, on the 1D PDF after the
square→spherical remap. Undo the blur there with `exp(2·blur·s²)` in reciprocal
space (note the factor **2** relative to the amplitude-space `exp(blur·stol²)`
in `sf_gemmi` — a Patterson is an intensity).

**Limitations (draft).** Orthogonal cells only; isotropic voxel; `margin_A`
(the atom guard-band around each block) must exceed the density cutoff radius.

## Visualization

The resulting diffuse scattering dataset can be viewed using [PDFViewer](https://github.com/aglie/DensityViewer) from the Yell package.

## Limitations (v1)

- **Anisotropic ADPs are supported** via the canonical IUCr U_cif → U_cart
  conversion in `sx_aniso_to_cart`. Heads-up for users on **gemmi ≤ 0.7.3**:
  `gemmi.mx_to_sx_structure` itself has a bug that skips the conversion for
  any cell with a 90° angle (hexagonal / trigonal / monoclinic). This
  package goes the other direction (sx → mx) with the correct formula, so
  it is unaffected — but downstream `mx_to_sx_structure` calls on the same
  cells are.
- **No symmetry expansion.** Configurations must already be in P1; there
  is no symmetry averaging.

## Acknowledgements

The code is based on work of Valentin Istomin, was tested by Johnathan Bulled and Cristian Ciomaga Hatnean.