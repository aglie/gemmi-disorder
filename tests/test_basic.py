"""
Basic correctness tests for gemmi-disorder.

These tests construct small toy structures in memory and verify:
- Grid.from_supercell produces the expected shape.
- DisorderedStructure converts to a gemmi.SmallStructure with the right
  cell, spacegroup and site count.
- sf_gemmi agrees with sf_gemmi_direct on a small grid (ground-truth check).
- average_diffuse over identical configurations gives diffuse ≈ 0.
- Non-zero aniso ADPs raise AnisoADPNotSupported.
"""

from pathlib import Path

import gemmi
import numpy as np
import pytest

from gemmi_disorder import (
    DisorderedStructure,
    Grid,
    average_diffuse,
    sf_gemmi,
    sf_gemmi_direct,
    sx_aniso_to_cart,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def two_carbon_structure(jitter: float = 0.0) -> DisorderedStructure:
    """Two carbon atoms in a 4Å cubic cell. `jitter` shifts the second atom
    along x for testing averaging.

    The cell is small enough that a modest grid (8³, crop=2 ⇒ padded 16³)
    resolves the carbon density (0.25 Å spacing) and the fast and direct
    paths agree to a few percent.
    """
    return DisorderedStructure(
        cell_parameters=(4.0, 4.0, 4.0, 90.0, 90.0, 90.0),
        atoms=[
            ("C", 0.0, 0.0, 0.0),
            ("C", 0.5 + jitter, 0.5, 0.5),
        ],
        supercell=(1, 1, 1),
    )


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def test_grid_from_supercell_shape():
    grid = Grid.from_supercell(supercell=(6, 6, 12), hkl_max=2)
    assert list(grid.no_pixels) == [24, 24, 48]
    np.testing.assert_allclose(grid.step_sizes, [1 / 6, 1 / 6, 1 / 12])
    np.testing.assert_allclose(grid.lower_limits, [-2, -2, -2])


def test_grid_from_supercell_tuple_hkl_max():
    grid = Grid.from_supercell(supercell=(4, 4, 4), hkl_max=(3, 2, 1))
    assert list(grid.no_pixels) == [24, 16, 8]
    np.testing.assert_allclose(grid.lower_limits, [-3, -2, -1])


# ---------------------------------------------------------------------------
# DisorderedStructure
# ---------------------------------------------------------------------------

def test_disordered_structure_to_small_structure():
    ds = two_carbon_structure()
    s = ds.to_small_structure()
    assert s.spacegroup_hm == "P 1"
    assert len(s.sites) == 2
    assert s.cell.a == ds.cell_parameters[0]
    assert s.sites[0].element.name == "C"
    np.testing.assert_allclose(
        [s.sites[1].fract.x, s.sites[1].fract.y, s.sites[1].fract.z],
        [0.5, 0.5, 0.5],
    )


def test_supercell_scaling():
    ds = DisorderedStructure(
        cell_parameters=(5.0, 5.0, 5.0, 90.0, 90.0, 90.0),
        atoms=[("C", 1.0, 2.0, 3.0)],  # in supercell coords
        supercell=(2, 4, 8),
    )
    s = ds.to_small_structure()
    assert s.cell.a == 10.0  # 5 * 2
    assert s.cell.b == 20.0  # 5 * 4
    assert s.cell.c == 40.0  # 5 * 8
    # atom at (1, 2, 3) in supercell coords becomes (0.5, 0.5, 0.375) fractional
    np.testing.assert_allclose(
        [s.sites[0].fract.x, s.sites[0].fract.y, s.sites[0].fract.z],
        [0.5, 0.5, 0.375],
    )


# ---------------------------------------------------------------------------
# Aniso ADP conversion
# ---------------------------------------------------------------------------

def _aniso_components(u):
    return (u.u11, u.u22, u.u33, u.u12, u.u13, u.u23)


def test_sx_aniso_to_cart_orthogonal_is_identity():
    """In an orthogonal cell U_cart == U_cif (gemmi's atom_to_site just copies)."""
    cell = gemmi.UnitCell(7.0, 8.0, 9.0, 90, 90, 90)
    u = gemmi.SMat33d(0.01, 0.02, 0.03, 0.004, 0.005, 0.006)
    out = sx_aniso_to_cart(u, cell)
    np.testing.assert_allclose(_aniso_components(out), _aniso_components(u))


def test_sx_aniso_to_cart_round_trip_non_orthogonal():
    """U_cart → U_cif (canonical formula) → U_cart (our function) should round-trip.

    We deliberately do NOT use `gemmi.mx_to_sx_structure` here — gemmi 0.7.3
    has a bug in `interop.hpp::atom_to_site` that copies U_cart verbatim into
    `site.aniso` whenever any cell angle is 90°. See
    docs/gemmi_interop_quirk.md.
    """
    cell = gemmi.UnitCell(7.0, 7.0, 6.0, 90, 90, 120)
    u_cart = gemmi.SMat33d(0.012, 0.013, 0.020, 0.0021, 0.0007, 0.0003)

    # Forward (mx → sx) via the IUCr-canonical formula:
    #   t = F U_cart F^T
    #   U_cif[i,j] = t[i,j] / (a*_i a*_j)
    F = np.array(cell.frac.mat)
    U_cart_mat = np.array([[u_cart.u11, u_cart.u12, u_cart.u13],
                           [u_cart.u12, u_cart.u22, u_cart.u23],
                           [u_cart.u13, u_cart.u23, u_cart.u33]], dtype=float)
    t = F @ U_cart_mat @ F.T
    rc = cell.reciprocal()
    v = np.array([1 / rc.a, 1 / rc.b, 1 / rc.c])
    U_cif_mat = t * np.outer(v, v)
    u_cif = gemmi.SMat33d(U_cif_mat[0, 0], U_cif_mat[1, 1], U_cif_mat[2, 2],
                          U_cif_mat[0, 1], U_cif_mat[0, 2], U_cif_mat[1, 2])

    # Inverse (sx → mx) via our function:
    recovered = sx_aniso_to_cart(u_cif, cell)
    np.testing.assert_allclose(_aniso_components(recovered),
                               _aniso_components(u_cart),
                               rtol=1e-6, atol=1e-10)


def test_sf_gemmi_with_aniso_matches_direct_orthogonal():
    """Single C atom at origin with an aniso ADP, orthogonal cell.
    Fast and direct paths should agree to within FFT discretization error."""
    s = gemmi.SmallStructure()
    s.name = "aniso_test"
    s.cell = gemmi.UnitCell(4.0, 4.0, 4.0, 90, 90, 90)
    s.spacegroup_hm = "P 1"

    site = gemmi.SmallStructure.Site()
    site.label = "C0"
    site.type_symbol = "C"
    site.element = gemmi.Element("C")
    site.fract = gemmi.Fractional(0.0, 0.0, 0.0)
    site.occ = 1.0
    # Aniso with non-trivial off-diagonal: oblate along z, tilted xy correlation
    site.aniso = gemmi.SMat33d(0.015, 0.015, 0.005, 0.003, 0.0, 0.0)
    s.add_site(site)

    grid = Grid(lower_limits=[-8, -8, -8], step_sizes=[1, 1, 1], no_pixels=[16, 16, 16])
    sf_fast = sf_gemmi(s, grid, blur=0.5, crop=2.0)
    sf_slow = sf_gemmi_direct(s, grid).values

    a_fast = np.abs(sf_fast)
    a_slow = np.abs(sf_slow)
    mask = a_slow > 0.1
    rel_err = np.abs(a_fast[mask] - a_slow[mask]) / a_slow[mask]
    assert np.max(rel_err) < 0.05, f"max rel err = {np.max(rel_err)}"


def test_sf_gemmi_with_aniso_matches_direct_hexagonal():
    """Same but in a hexagonal cell — exercises the U_cif → U_cart conversion."""
    s = gemmi.SmallStructure()
    s.name = "aniso_hex"
    s.cell = gemmi.UnitCell(5.0, 5.0, 5.0, 90, 90, 120)
    s.spacegroup_hm = "P 1"

    site = gemmi.SmallStructure.Site()
    site.label = "C0"
    site.type_symbol = "C"
    site.element = gemmi.Element("C")
    site.fract = gemmi.Fractional(0.1, 0.2, 0.3)
    site.occ = 1.0
    site.aniso = gemmi.SMat33d(0.012, 0.014, 0.008, 0.004, 0.001, 0.002)
    s.add_site(site)

    grid = Grid(lower_limits=[-6, -6, -6], step_sizes=[1, 1, 1], no_pixels=[12, 12, 12])
    sf_fast = sf_gemmi(s, grid, blur=0.5, crop=2.0)
    sf_slow = sf_gemmi_direct(s, grid).values

    a_fast = np.abs(sf_fast)
    a_slow = np.abs(sf_slow)
    mask = a_slow > 0.1
    rel_err = np.abs(a_fast[mask] - a_slow[mask]) / a_slow[mask]
    assert np.max(rel_err) < 0.07, f"max rel err = {np.max(rel_err)}"


def test_dy467_with_real_aniso():
    """Load the Dy467 CIF (real anisotropic ADPs, hexagonal P63/m → P1 expansion)
    and compare a handful of low-order reflections between sf_gemmi and
    sf_gemmi_direct.

    NB: when expanding the P63/m structure to P1, we must build a fresh
    `gemmi.UnitCell` from scratch and not just `p1.cell = small.cell` —
    `UnitCell.images` (the list of symmetry-mate translations) survives a
    plain assignment and `calculate_sf_from_small_structure` silently re-applies
    them, multiplying F(hkl) by the order of the original spacegroup."""
    cif_path = Path(__file__).parent / "Dy467.cif"
    if not cif_path.exists():
        pytest.skip("Dy467.cif not available")

    small = gemmi.read_small_structure(str(cif_path))
    small.change_occupancies_to_crystallographic()

    p1 = gemmi.SmallStructure()
    p1.cell = gemmi.UnitCell(small.cell.a, small.cell.b, small.cell.c,
                             small.cell.alpha, small.cell.beta, small.cell.gamma)
    p1.spacegroup_hm = "P 1"
    for s in small.get_all_unit_cell_sites():
        p1.add_site(s)

    # 4³ output grid with crop=8 → padded 32³ → 0.29 Å spacing, enough to
    # resolve Dy density. (Direct-path slow loop scales with grid size, so we
    # keep the output grid small and oversample via `crop` instead.)
    grid = Grid(lower_limits=[-2, -2, -2], step_sizes=[1, 1, 1], no_pixels=[4, 4, 4])
    sf_fast = sf_gemmi(p1, grid, blur=0.5, crop=8.0)
    sf_slow = sf_gemmi_direct(p1, grid).values

    a_fast = np.abs(sf_fast)
    a_slow = np.abs(sf_slow)

    # F(000) must match the expected total electron count.
    centre = tuple(g // 2 for g in grid.no_pixels)
    z_total = sum(s.occ * s.element.atomic_number for s in p1.sites)
    np.testing.assert_allclose(np.abs(sf_slow[centre]), z_total, rtol=0.01)
    np.testing.assert_allclose(np.abs(sf_fast[centre]), z_total, rtol=0.03)

    mask = a_slow > 5
    rel_err = np.abs(a_fast[mask] - a_slow[mask]) / a_slow[mask]
    assert np.max(rel_err) < 0.10, f"Dy467 max rel err = {np.max(rel_err)}"


# ---------------------------------------------------------------------------
# sf_gemmi vs sf_gemmi_direct on a small grid
# ---------------------------------------------------------------------------

def test_sf_gemmi_matches_direct_on_small_grid():
    """The fast path (density + FFT + deblur) should agree with the direct
    Miller-by-Miller calculation to within a few percent on a small grid."""
    s = two_carbon_structure().to_small_structure()

    # 16³ on a 4Å cell with crop=2 gives a 32³ padded grid at 0.125Å spacing
    # — fine enough that the IT92 form-factor Gaussians (narrowest b≈0.57 for C)
    # are resolved. Convergence is essentially exact at n=32, ~2% at n=16.
    grid = Grid(
        lower_limits=[-8, -8, -8],
        step_sizes=[1, 1, 1],
        no_pixels=[16, 16, 16],
    )

    sf_fast = sf_gemmi(s, grid, blur=0.5, crop=2.0)
    sf_slow = sf_gemmi_direct(s, grid).values

    a_fast = np.abs(sf_fast)
    a_slow = np.abs(sf_slow)

    # F(000) absolute check: must equal sum of atomic Zs (≈ 2 × 6 = 12).
    centre = tuple(g // 2 for g in grid.no_pixels)
    np.testing.assert_allclose(np.abs(sf_fast[centre]), 12.0, rtol=0.01)
    np.testing.assert_allclose(np.abs(sf_slow[centre]), 12.0, rtol=0.01)

    # Compare reflections with non-negligible amplitude.
    mask = a_slow > 0.5
    rel_err = np.abs(a_fast[mask] - a_slow[mask]) / a_slow[mask]
    assert np.max(rel_err) < 0.05, f"max relative error = {np.max(rel_err)}"


# ---------------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------------

def test_average_diffuse_identical_configs_is_zero():
    """If every configuration is identical, ⟨I⟩ = |⟨F⟩|² and diffuse ≈ 0."""
    structures = [two_carbon_structure() for _ in range(3)]
    grid = Grid(lower_limits=[-2, -2, -2], step_sizes=[1, 1, 1], no_pixels=[4, 4, 4])

    result = average_diffuse(structures, grid, blur=0.5, crop=2.0)
    assert result.n_configs == 3
    assert result.average_F.shape == tuple(grid.no_pixels)
    np.testing.assert_allclose(result.diffuse, 0, atol=1e-6)


def test_average_diffuse_jittered_configs_nonzero_diffuse():
    """If configurations differ, the diffuse map should be non-zero somewhere."""
    structures = [two_carbon_structure(jitter=j) for j in (-0.05, 0.0, 0.05)]
    grid = Grid(lower_limits=[-2, -2, -2], step_sizes=[1, 1, 1], no_pixels=[4, 4, 4])

    result = average_diffuse(structures, grid, blur=0.5, crop=2.0)
    assert np.max(result.diffuse) > 1e-3
    # Energy conservation sanity: ⟨I⟩ >= |⟨F⟩|² everywhere (modulo num. noise).
    assert np.all(result.diffuse > -1e-6)


def test_keep_per_snapshot():
    structures = [two_carbon_structure() for _ in range(2)]
    grid = Grid(lower_limits=[-1, -1, -1], step_sizes=[1, 1, 1], no_pixels=[2, 2, 2])

    result = average_diffuse(structures, grid, keep_per_snapshot=True, blur=0.5, crop=2.0)
    assert result.per_snapshot_F is not None
    assert len(result.per_snapshot_F) == 2
    np.testing.assert_allclose(result.per_snapshot_F[0], result.per_snapshot_F[1])
