"""
Correctness tests for the tiled 3D-Patterson (`gemmi_disorder.pdf`).

The key invariant is that the overlap-save decomposition reproduces the
autocorrelation of the *whole* (partitioned) density — i.e. it is exact, not
approximate — for `|r| ≤ r_max` once the block side is ≥ the lag window. We
check that against a brute-force full-FFT reference, and separately check that
the disk-budgeted block cache (evict + rebuild) does not perturb the result.

These use a small FCC Cu supercell so the grids stay tiny and the tests run in
well under a second.
"""

import numpy as np
import pytest

from gemmi_disorder import DisorderedStructure, tiled_patterson
from gemmi_disorder import pdf as P


A0 = 3.61  # Å, FCC Cu conventional cell


def fcc_cu(nrep: int) -> DisorderedStructure:
    """Conventional cubic FCC Cu, `nrep`³ supercell (4·nrep³ atoms)."""
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    atoms = [
        ("Cu", ix + bx, iy + by, iz + bz)
        for ix in range(nrep) for iy in range(nrep) for iz in range(nrep)
        for (bx, by, bz) in basis
    ]
    return DisorderedStructure((A0, A0, A0, 90, 90, 90), atoms,
                               supercell=(nrep, nrep, nrep))


def _stitched_reference(structure, q_max, r_max, block_vox, margin_A=3.0, blur=0.5):
    """Full (non-tiled) Patterson from the SAME block densities, via one big FFT.

    Rebuilds every block with the module's own builder, stitches them into the
    full density grid, and autocorrelates it with zero padding (no wraparound).
    This isolates the overlap-save bookkeeping from the density build: both
    sides see byte-identical `ρ_I`, so any disagreement is a lag/index bug.
    """
    dr = np.pi / q_max
    M = int(round(r_max / dr))
    margin_vox = int(np.ceil(margin_A / dr))
    builder = P._make_block_builder(structure, block_vox, dr, margin_vox, blur, 0.0)
    cell = structure.cell
    lengths = np.array([cell.a, cell.b, cell.c])
    nb = np.ceil(np.ceil(lengths / dr) / block_vox).astype(int)
    full = np.zeros(tuple(nb * block_vox), dtype=np.float64)
    for I in np.ndindex(*nb):
        sl = tuple(slice(I[a] * block_vox, (I[a] + 1) * block_vox) for a in range(3))
        full[sl] = builder(I)
    return P.reference_patterson_from_density(full, M), M


# ---------------------------------------------------------------------------
# API / shape
# ---------------------------------------------------------------------------

def test_patterson_window_shape_and_metadata():
    ss = fcc_cu(6).to_small_structure()
    q_max, r_max = 10.0, 6.0
    pw = tiled_patterson(ss, q_max, r_max, blur=0.5, apply_dv=False)
    dr = np.pi / q_max
    M = int(round(r_max / dr))
    assert pw.data.shape == (2 * M + 1,) * 3
    assert pw.M == M
    np.testing.assert_allclose(pw.r_step, dr)
    assert pw.blur == 0.5                       # left in, to be removed downstream
    # Patterson origin is the global maximum (self-overlap of the density).
    assert np.argmax(pw.data) == pw.data.size // 2


# ---------------------------------------------------------------------------
# Exactness of the overlap-save decomposition
# ---------------------------------------------------------------------------

def test_tiled_matches_full_fft_reference():
    """block_vox > M ⇒ direct-neighbour-only is exact on every voxel."""
    ss = fcc_cu(8).to_small_structure()
    q_max, r_max = 10.0, 6.0
    M = int(round(r_max / (np.pi / q_max)))
    block_vox = M + 2                            # strictly larger than the window

    pw = tiled_patterson(ss, q_max, r_max, block_vox=block_vox,
                         blur=0.5, apply_dv=False)
    ref, _ = _stitched_reference(ss, q_max, r_max, block_vox)

    scale = np.abs(ref).max()
    assert np.abs(pw.data - ref).max() / scale < 1e-10


def test_tiled_block_equals_window_interior_exact():
    """With block_vox == M the interior is still exact; only the |r|=M shell can
    miss the (negligible) two-blocks-away contribution."""
    ss = fcc_cu(8).to_small_structure()
    q_max, r_max = 10.0, 6.0
    M = int(round(r_max / (np.pi / q_max)))

    pw = tiled_patterson(ss, q_max, r_max, block_vox=M, blur=0.5, apply_dv=False)
    ref, _ = _stitched_reference(ss, q_max, r_max, M)

    scale = np.abs(ref).max()
    cheby = np.abs(np.indices(pw.data.shape) - M).max(axis=0)   # |r| in voxels
    interior = cheby < M
    assert np.abs(pw.data[interior] - ref[interior]).max() / scale < 1e-10


# ---------------------------------------------------------------------------
# Disk-budgeted cache: eviction + rebuild must be lossless
# ---------------------------------------------------------------------------

def test_disk_eviction_reproduces_result(tmp_path):
    ss = fcc_cu(8).to_small_structure()
    q_max, r_max = 10.0, 6.0
    block_vox = int(round(r_max / (np.pi / q_max))) + 2

    roomy = tiled_patterson(ss, q_max, r_max, block_vox=block_vox,
                            blur=0.5, apply_dv=False,
                            disk_budget_bytes=8 * 1024 ** 3, mem_blocks=6)
    # 2 MB budget + a single in-RAM block forces constant eviction and rebuild.
    tight = tiled_patterson(ss, q_max, r_max, block_vox=block_vox,
                            blur=0.5, apply_dv=False,
                            cache_dir=str(tmp_path / "blk"),
                            disk_budget_bytes=2 * 1024 ** 2, mem_blocks=1)
    np.testing.assert_array_equal(roomy.data, tight.data)


def test_block_cache_rebuilds_when_disk_disabled():
    """disk_budget_bytes=0 disables disk; evicted blocks are rebuilt from atoms."""
    built = {}

    def builder(I):
        built[I] = built.get(I, 0) + 1
        return np.zeros((3, 3, 3), dtype=np.float32)

    cache = P.BlockDensityCache(builder, cache_dir=".unused_cache",
                                disk_budget_bytes=0, mem_blocks=1)
    cache.get((0, 0, 0))
    cache.get((1, 0, 0))          # evicts (0,0,0) from RAM; no disk to spill to
    cache.get((0, 0, 0))          # must rebuild
    cache.close()
    assert built[(0, 0, 0)] == 2
    assert cache.n_built == 3


# ---------------------------------------------------------------------------
# _axis_slices bookkeeping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("off,src,dst,expect", [
    (0, 4, 4, (slice(0, 4), slice(0, 4))),      # aligned
    (2, 4, 4, (slice(0, 2), slice(2, 4))),      # positive shift, right-clipped
    (-2, 4, 4, (slice(2, 4), slice(0, 2))),     # negative shift, left-clipped
    (10, 4, 4, None),                           # no overlap (past the end)
    (-10, 4, 4, None),                          # no overlap (before the start)
])
def test_axis_slices(off, src, dst, expect):
    assert P._axis_slices(off, src, dst) == expect
