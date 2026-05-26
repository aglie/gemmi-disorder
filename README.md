# gemmi-disorder

A small wrapper around [gemmi](https://github.com/project-gemmi/gemmi) for
computing **diffuse scattering** from collections of disordered atomic
configurations. The calculation is done by averaging structure factors and
intensities over many realisations of a disordered structure to obtain:

- ⟨F⟩ — average complex structure factor (the Bragg peaks).
- ⟨I⟩ — total scattering.
- ⟨I⟩ − |⟨F⟩|² — the diffuse scattering map.

Built on gemmi's `DensityCalculatorX` fast FFT-based algorithm (the "fast path"). A slow
but exact `sf_gemmi_direct` which calculates structure factors by direct summation is also included for validation.

## Install

```bash
pip install -e .
# optional ASE bridge for MC / ORB workflows:
pip install -e .[ase]
```

## Usage

### From CIF files

Each input CIF must be one realised disordered configuration in P1, with a
unit cell that is the `supercell`-times-supercell of the underlying motif.
That is: if `supercell = (nx, ny, nz)` and the underlying motif has cell
`(a, b, c, α, β, γ)`, then every CIF should have cell
`(a·nx, b·ny, c·nz, α, β, γ)` and contain the full supercell's worth of
atoms. The grid below is built around that supercell.

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

## Limitations (v1)

- **Anisotropic ADPs are supported** via the canonical IUCr U_cif → U_cart
  conversion in `sx_aniso_to_cart`. Heads-up for users on **gemmi ≤ 0.7.3**:
  `gemmi.mx_to_sx_structure` itself has a bug that skips the conversion for
  any cell with a 90° angle (hexagonal / trigonal / monoclinic). This
  package goes the other direction (sx → mx) with the correct formula, so
  it is unaffected — but downstream `mx_to_sx_structure` calls on the same
  cells are. See `docs/gemmi_interop_quirk.md`.
- **No symmetry expansion.** Configurations must already be in P1; there
  is no symmetry averaging.
- **All snapshots in memory.** v1 takes a Python list; streaming and
  checkpointing can be added later.

## Acknowledgements

The fast path and Yell HDF5 writer are taken verbatim from the
`CalculateScattering.py` files used in the Dy467 disorder analysis and the
Seminar 7 disordered-materials teaching code; see
`docs/CALCULATE_SCATTERING_COMPARISON.md` in the jax-ftl repo for the
provenance and decisions behind this consolidation.
