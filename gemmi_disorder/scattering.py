"""
Structure factor calculation using gemmi.

Two paths are provided:

- `sf_gemmi(structure, grid, blur, crop)` — the fast path. Builds the electron
  density on a 3D grid via `gemmi.DensityCalculatorX`, FFTs it, then undoes
  the blur using gemmi's own reciprocal-space multiplier. Use this for
  production runs over many disordered configurations.

- `sf_gemmi_direct(structure, grid)` — the slow but exact path: a triple loop
  calling `gemmi.StructureFactorCalculatorX.calculate_sf_from_small_structure`
  at every grid point. Use it as a ground-truth check on small grids.

The structure factor formula is the standard one for small molecules; both
functions take a `gemmi.SmallStructure` in P1 (no symmetry expansion is done).

ADP handling
------------
`gemmi.SmallStructure.AtomSite.aniso` (the CIF/small-mol convention) and
`gemmi.Atom.aniso` (the macromolecular Cartesian convention) store U in
different bases. We convert sx → mx in `sx_aniso_to_cart`, inverting gemmi's
own `interop.atom_to_site` formula:

    site.aniso[i,j] = (F · U_cart · Fᵀ)[i,j] · (1/a*_i) · (1/a*_j)

so to go the other way:

    t[i,j] = site.aniso[i,j] · a*_i · a*_j
    U_cart = O · t · Oᵀ                    (since O = F⁻¹)

For orthogonal cells gemmi's `atom_to_site` just copies the U components,
so we do the same here as a shortcut.
"""

from typing import Tuple

import gemmi
import numpy as np
from numpy.typing import NDArray

from .grid import Grid, StructureFactors


def _is_orthogonal(cell: gemmi.UnitCell) -> bool:
    return cell.alpha == 90.0 and cell.beta == 90.0 and cell.gamma == 90.0


def sx_aniso_to_cart(u_cif: gemmi.SMat33d, cell: gemmi.UnitCell) -> gemmi.SMat33d:
    """Convert a U tensor from CIF/small-mol convention to Cartesian (mx).

    Inverts the formula used by `gemmi::atom_to_site` in `interop.hpp`:

        site.aniso[i,j] = (F U_cart Fᵀ)[i,j] · (1/a*_i) · (1/a*_j)

    For orthogonal cells the conversion is the identity (and we return a
    shallow copy so the output is independent from the input).
    """
    if _is_orthogonal(cell):
        return gemmi.SMat33d(u_cif.u11, u_cif.u22, u_cif.u33,
                             u_cif.u12, u_cif.u13, u_cif.u23)

    rc = cell.reciprocal()
    ar, br, cr = rc.a, rc.b, rc.c

    # Scale each component by a*_i · a*_j to undo the v[i]·v[j] step.
    t = gemmi.SMat33d(
        u_cif.u11 * ar * ar,
        u_cif.u22 * br * br,
        u_cif.u33 * cr * cr,
        u_cif.u12 * ar * br,
        u_cif.u13 * ar * cr,
        u_cif.u23 * br * cr,
    )
    # Then transform by the orthogonalization matrix O: U_cart = O t Oᵀ.
    return t.transformed_by(cell.orth.mat)


def sx_to_mx_structure(inp: gemmi.SmallStructure, b_iso: float = 0.0) -> gemmi.Structure:
    """Convert small-molecule structure to macromolecular structure.

    `gemmi.DensityCalculatorX` works on `gemmi.Structure` rather than
    `SmallStructure`; this helper wraps the sites of a small-mol structure
    into a single chain/residue/model so the density calculator can be used.

    Anisotropic U is propagated by converting from the CIF/sx basis to the
    Cartesian/mx basis via `sx_aniso_to_cart`. The (also isotropic)
    `site.u_iso` would be the natural per-atom isotropic ADP, but here we
    use a single `b_iso` for all atoms — that matches the existing
    CalculateScattering pipeline and keeps the call site explicit.

    Parameters
    ----------
    inp: gemmi.SmallStructure
        Input structure (P1 expected; no symmetry is applied here).
    b_iso: float
        Isotropic B-factor assigned to every atom. Default 0.0 matches the
        Dy467 production pipeline. Seminar-7 callers used 0.5 as a numerical
        stabiliser; pass it explicitly if you want that behaviour.
    """
    out = gemmi.Structure()
    out.name = inp.name
    out.cell = inp.cell
    out.spacegroup_hm = inp.spacegroup_hm
    model = gemmi.Model('1')
    chain = gemmi.Chain('A')
    res = gemmi.Residue()
    for i, site in enumerate(inp.sites):
        at = gemmi.Atom()
        at.name = f"{site.element.name}{i}"
        at.element = site.element
        at.pos = site.orth(inp.cell)
        at.b_iso = b_iso
        at.occ = site.occ
        if site.aniso.nonzero():
            u = sx_aniso_to_cart(site.aniso, inp.cell)
            at.aniso = gemmi.SMat33f(u.u11, u.u22, u.u33, u.u12, u.u13, u.u23)
        res.add_atom(at)
    chain.add_residue(res)
    model.add_chain(chain)
    out.add_model(model)
    return out


def sf_gemmi_direct(structure: gemmi.SmallStructure, grid: Grid) -> StructureFactors:
    """Structure factors by direct calculation at every Miller index in `grid`.

    Slow (one gemmi call per grid point) but exact. Use for validation of
    `sf_gemmi` on small grids.
    """
    calc = gemmi.StructureFactorCalculatorX(structure.cell)

    res = np.zeros(grid.no_pixels, dtype=complex)

    for hi in range(grid.no_pixels[0]):
        for ki in range(grid.no_pixels[1]):
            for li in range(grid.no_pixels[2]):
                indices = [hi, ki, li]
                h, k, l = [int(round(grid.lower_limits[i] + grid.step_sizes[i] * indices[i])) for i in range(3)]

                res[hi, ki, li] = calc.calculate_sf_from_small_structure(structure, [h, k, l])

    return StructureFactors(values=res, grid=grid)


def sf_gemmi(structure: gemmi.SmallStructure,
             grid: Grid,
             blur: float = 0.01,
             crop: float = 1.8,
             b_iso: float = 0.0) -> NDArray[np.complex128]:
    """Calculate structure factors via density-on-grid + FFT (fast path).

    Algorithm:
        1. Pad the grid by `crop` to oversample the density before FFT.
        2. Convert the small-molecule structure to a macromolecular one and
           place its electron density on the padded grid using gemmi's
           DensityCalculatorX, with `dencalc.blur = blur` (extra B-factor in Å²).
        3. FFT, fftshift, slice back to `grid.no_pixels`.
        4. Undo the blur: multiply each F(h,k,l) by `exp(blur * stol²)`.
           This is exactly what `dencalc.reciprocal_space_multiplier(stol²)`
           computes; we apply the formula directly to keep the per-grid-point
           operation vectorised.

    Parameters
    ----------
    structure: gemmi.SmallStructure
        Input structure (P1, aniso==0).
    grid: Grid
        Output grid in reciprocal space.
    blur: float
        Extra isotropic B-factor (Å²) added to every atom before density
        placement. Larger blur ⇒ smoother density ⇒ fewer FFT artefacts but
        larger inverse correction. 0.01 matches Dy467 production.
    crop: float
        Oversampling factor (≥1) used to pad the density grid before FFT.
    b_iso: float
        Base isotropic B-factor assigned to every atom in `sx_to_mx_structure`.
        Default 0.0 matches Dy467; set to 0.5 to reproduce Seminar 7 results.

    Returns
    -------
    Complex structure factor array of shape `grid.no_pixels`.
    """
    padded_grid = grid.pad(crop)

    str_mx = sx_to_mx_structure(structure, b_iso=b_iso)
    dencalc = gemmi.DensityCalculatorX()
    # `structure.spacegroup` is populated when reading a CIF but may be None
    # when the structure was built in memory; fall back to looking it up
    # from the HM symbol (defaulting to P1, which is what this package assumes).
    sg = structure.spacegroup or gemmi.find_spacegroup_by_name(structure.spacegroup_hm or "P 1")
    dencalc.grid.spacegroup = sg
    dencalc.grid.unit_cell = structure.cell
    dencalc.grid.set_size(*padded_grid.no_pixels)
    dencalc.blur = blur

    dencalc.put_model_density_on_grid(str_mx[0])

    # Normalize by dV = V / N_padded so the FFT amplitude matches the
    # crystallographic structure-factor convention (F(000) = sum of atomic Zs).
    # The Dy467/Seminar 7 versions of this function omit this factor; downstream
    # diffuse-scattering code only depended on relative magnitudes so the
    # omission was invisible. Including it here makes the result directly
    # comparable to `sf_gemmi_direct` and to gemmi's `transform_map_to_f_phi`.
    dV = float(structure.cell.volume) / np.prod(padded_grid.no_pixels)
    sf_grid = np.fft.fftn(np.array(dencalc.grid)) * dV
    sf_grid = np.fft.fftshift(sf_grid)

    padding = grid.padding(crop)
    sf_grid = sf_grid[padding[0]:-padding[0], padding[1]:-padding[1], padding[2]:-padding[2]]

    # Deblur: undo the extra isotropic B-factor introduced by `dencalc.blur`.
    # gemmi's reciprocal_space_multiplier(stol²) returns exp(blur * stol²);
    # we vectorise that across the whole grid using the reciprocal metric tensor.
    inv_metric = np.array(structure.cell.reciprocal_metric_tensor().as_mat33())
    q_vectors = generate_q_vectors(grid)
    stol_squared = calculate_stol_squared(q_vectors, inv_metric)
    sf_grid = sf_grid * np.exp(blur * stol_squared)

    return sf_grid


def get_form_factors(element: str) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Get form factor coefficients a, b, c for given element"""
    coef = gemmi.Element(element).it92
    return np.array(coef.a), np.array(coef.b), coef.c


def generate_q_vectors(grid: Grid) -> NDArray[np.float64]:
    """Generate q-vectors (Miller indices in r.l.u.) for each grid point."""
    qx, qy, qz = [np.arange(grid.no_pixels[d]) * grid.step_sizes[d] + grid.lower_limits[d] for d in range(3)]

    qxg, qyg, qzg = np.meshgrid(qx, qy, qz, indexing='ij')
    return np.stack([qxg, qyg, qzg], axis=-1)


def calculate_stol_squared(q_vectors: NDArray[np.float64],
                           inv_metric: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute (sin θ / λ)² = (1/d²)/4 at each grid point.

    `q_vectors` carry Miller indices in r.l.u. (shape (..., 3)); `inv_metric`
    is the reciprocal metric tensor G* (3×3) such that hᵀ G* h = 1/d².
    """
    return np.einsum('ijkl,lm,ijkm->ijk', q_vectors, inv_metric, q_vectors) / 4


