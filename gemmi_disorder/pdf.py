"""
Tiled 3D-Patterson (3D-PDF) evaluation — draft.

Computes the density autocorrelation

    P(r) = Σ_x ρ(x) ρ(x + r)          for |r_α| ≤ r_max

directly in real space, on a window small enough to hold in memory, without
ever materialising the full-box reciprocal cube that `sf_gemmi` needs. See
`docs/tiled_3dpdf_proposal.md` for the reasoning; this module is the
`tiled_patterson` piece of that proposal.

What this does and (deliberately) does NOT do
---------------------------------------------
- It returns the *raw, blurred* Patterson window. **No deblur and no window
  taper are applied here.** Both are cheap point-wise corrections that are far
  better done at the very end, on the 1D PDF after the square→spherical
  remapping — applying them in 3D would only fight that remapping. `blur` still
  enters (the density is built with it for FFT stability); it is simply left in
  the output for the caller to remove later. See `PattersonWindow.blur`.

- Overlap-save decomposition (§ "Algorithm" in the proposal). The box is cut
  into cubic blocks of side `block_vox ≥ M` (M = r_max in voxels). With the
  block at least as large as the lag window, a block's autocorrelation can only
  reach into `|r| ≤ r_max` from **its 26 direct neighbours (and itself)** — all
  farther block pairs are skipped. We assume (per the brief) that the output
  window fits comfortably in RAM several times over, so we never need a coarser
  decomposition than direct-neighbour.

- In 3D the streaming "front" of block densities is large, so blocks are cached
  on disk and recomputed on eviction. `disk_budget_bytes` caps the cache; when
  it is exceeded the least-recently-used block files are deleted and rebuilt
  from the atom list the next time they are needed.

Assumptions / limitations (draft)
---------------------------------
- Orthogonal cell only (α=β=γ=90°). Cartesian blocks are then axis-aligned with
  fractional coordinates, which keeps the bookkeeping honest. Non-orthogonal
  cells would need block shapes in the crystal frame — out of scope for v0.
- Isotropic voxel `Δr = π/q_max` on all three axes; cubic blocks in Å.
- Absolute scale follows `sf_gemmi`'s `dV = V/N` convention, applied once per
  density factor (so `dV²` overall); toggle with `apply_dv`.
"""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import gemmi
import numpy as np
from numpy.typing import NDArray

from .scattering import _make_density_calculator, sx_to_mx_structure

# The 27 block offsets (including self) that a block can correlate with when
# block_vox ≥ M. Order is irrelevant; each *ordered* pair is visited once so the
# sum over neighbours reproduces Σ_{I,J adjacent} with no double counting.
_NEIGHBOUR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
]


@dataclass
class PattersonWindow:
    """Raw (blurred, un-windowed) 3D-Patterson on |r_α| ≤ r_max.

    Attributes
    ----------
    data : (W, W, W) float array, W = 2*M + 1
        P(r); `data[M, M, M]` is the r = 0 origin. Blur is *not* removed.
    r_step : float
        Voxel size Δr in Å (= π / q_max).
    r_max : float
        Half-window in Å (= M · Δr).
    blur : float
        Extra isotropic B-factor left in the density (Å²) — deblur later with
        exp(2·blur·s²) in reciprocal space, or the real-space equivalent, once
        you are on the 1D PDF.
    cell : gemmi.UnitCell
        The (supercell) cell the structure lived in, for downstream metadata.
    """

    data: NDArray[np.float64]
    r_step: float
    r_max: float
    blur: float
    cell: gemmi.UnitCell

    @property
    def M(self) -> int:
        return self.data.shape[0] // 2


# ---------------------------------------------------------------------------
# overlap-save bookkeeping
# ---------------------------------------------------------------------------
def _axis_slices(off: int, src_len: int, dst_len: int) -> Optional[Tuple[slice, slice]]:
    """Slices for a shifted add `dst[off + k] += src[k]`, clipped to both ends.

    Returns (src_slice, dst_slice) or None if there is no overlap.
    """
    d0 = max(0, off)
    d1 = min(dst_len, off + src_len)
    if d1 <= d0:
        return None
    return slice(d0 - off, d1 - off), slice(d0, d1)


def _accumulate_pair(pw: NDArray[np.float64],
                     xcc: NDArray[np.float64],
                     offset: Tuple[int, int, int]) -> None:
    """Add one block-pair correlation `xcc` into the window `pw`.

    `xcc` is the fftshift-ed linear cross-correlation of blocks (I, J = I + D),
    indexed so `xcc[k]` is the correlation at intra-pair lag `t = k - B`
    (k ∈ [0, 2B)). The global lag is `r = D·B + t`, so with window index
    `r + M` the destination is `off + k` where `off = D·B + (M - B)` per axis
    (see proposal, "The correct convolution chain").
    """
    B = xcc.shape[0] // 2               # block size (xcc has side 2B)
    M = pw.shape[0] // 2
    slabs = []
    for a in range(3):
        off = offset[a] * B + (M - B)
        sl = _axis_slices(off, xcc.shape[a], pw.shape[a])
        if sl is None:
            return
        slabs.append(sl)
    src = (slabs[0][0], slabs[1][0], slabs[2][0])
    dst = (slabs[0][1], slabs[1][1], slabs[2][1])
    pw[dst] += xcc[src]


# ---------------------------------------------------------------------------
# disk-budgeted block-density cache
# ---------------------------------------------------------------------------
class BlockDensityCache:
    """LRU cache of per-block densities, backed by disk with a byte budget.

    A block's density is expensive to build (atom placement) but cheap to store
    (B³ float32). We keep a few in RAM (`mem_blocks`), spill the rest to `.npy`
    files under `cache_dir`, and once the on-disk bytes exceed
    `disk_budget_bytes` we delete the least-recently-used files. A later request
    for an evicted block simply rebuilds it from the atom list — this is the
    "kick some portions and recalculate as needed" knob.
    """

    def __init__(self,
                 builder: Callable[[Tuple[int, int, int]], NDArray[np.float32]],
                 cache_dir: Path,
                 disk_budget_bytes: int,
                 mem_blocks: int = 4):
        self.builder = builder
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.disk_budget_bytes = int(disk_budget_bytes)
        self.mem_blocks = int(mem_blocks)

        self._mem: "OrderedDict[Tuple[int,int,int], NDArray[np.float32]]" = OrderedDict()
        self._disk: "OrderedDict[Tuple[int,int,int], Tuple[Path,int]]" = OrderedDict()
        self._disk_bytes = 0
        # counters, handy for tuning / tests
        self.n_built = 0
        self.n_mem_hits = 0
        self.n_disk_hits = 0

    def _path(self, I: Tuple[int, int, int]) -> Path:
        return self.cache_dir / f"blk_{I[0]}_{I[1]}_{I[2]}.npy"

    def get(self, I: Tuple[int, int, int]) -> NDArray[np.float32]:
        if I in self._mem:
            self._mem.move_to_end(I)
            self.n_mem_hits += 1
            return self._mem[I]
        if I in self._disk:
            path, nbytes = self._disk.pop(I)
            arr = np.load(path)
            self.n_disk_hits += 1
            self._disk_bytes -= nbytes
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            self._insert_mem(I, arr)
            return arr
        arr = self.builder(I)
        self.n_built += 1
        self._insert_mem(I, arr)
        return arr

    def _insert_mem(self, I, arr) -> None:
        self._mem[I] = arr
        self._mem.move_to_end(I)
        while len(self._mem) > self.mem_blocks:
            old_I, old_arr = self._mem.popitem(last=False)
            self._spill_to_disk(old_I, old_arr)

    def _spill_to_disk(self, I, arr) -> None:
        if self.disk_budget_bytes <= 0:
            return                       # disk disabled → just drop (rebuild later)
        nbytes = arr.nbytes
        path = self._path(I)
        np.save(path, arr)
        self._disk[I] = (path, nbytes)
        self._disk_bytes += nbytes
        self._evict_disk()

    def _evict_disk(self) -> None:
        while self._disk_bytes > self.disk_budget_bytes and self._disk:
            _, (path, nbytes) = self._disk.popitem(last=False)
            self._disk_bytes -= nbytes
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def close(self) -> None:
        """Remove the whole cache directory."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# per-block density builder (gemmi)
# ---------------------------------------------------------------------------
def _require_orthogonal(cell: gemmi.UnitCell) -> None:
    if not (cell.alpha == 90.0 and cell.beta == 90.0 and cell.gamma == 90.0):
        raise NotImplementedError(
            "tiled_patterson (draft) supports orthogonal cells only; "
            f"got angles ({cell.alpha}, {cell.beta}, {cell.gamma})."
        )


def _make_block_builder(structure: gemmi.SmallStructure,
                        block_vox: int,
                        dr: float,
                        margin_vox: int,
                        blur: float,
                        b_iso: float,
                        scattering: str = "xray") -> Callable[[Tuple[int, int, int]], NDArray[np.float32]]:
    """Return a function I -> ρ_I (B³ float32), the true density masked to block I.

    ρ_I is *not* "the density of block I's atoms" but "the total density,
    restricted to block I's voxels". We therefore place every atom whose
    Gaussian can reach block I — those within `margin` of the block — and crop
    the result to the central B³. Gaussian tails from an atom near a boundary
    thus appear in *both* adjacent blocks, which is correct: the blocks
    partition the voxels, not the atoms, so Σ_I ρ_I = ρ exactly.
    """
    cell = structure.cell
    _require_orthogonal(cell)
    lengths = np.array([cell.a, cell.b, cell.c], dtype=float)

    # Cartesian positions and elements, extracted once.
    cart = np.array([[s.orth(cell).x, s.orth(cell).y, s.orth(cell).z]
                     for s in structure.sites], dtype=float)
    elements = [s.element.name for s in structure.sites]

    G = block_vox + 2 * margin_vox       # padded mini-grid side (voxels)
    mini_len = G * dr                    # padded mini-cell side (Å)
    margin_A = margin_vox * dr
    block_A = block_vox * dr
    sg_p1 = gemmi.find_spacegroup_by_name("P 1")

    def build(I: Tuple[int, int, int]) -> NDArray[np.float32]:
        origin = np.array(I, dtype=float) * block_A          # block lower corner, Å
        lo = origin - margin_A
        hi = origin + block_A + margin_A
        mask = np.all((cart >= lo) & (cart < hi), axis=1)
        idx = np.nonzero(mask)[0]

        mini = gemmi.SmallStructure()
        mini.cell = gemmi.UnitCell(mini_len, mini_len, mini_len, 90.0, 90.0, 90.0)
        mini.spacegroup_hm = "P 1"
        for j in idx:
            site = gemmi.SmallStructure.Site()
            el = elements[j]
            site.label = el
            site.type_symbol = el
            site.element = gemmi.Element(el)
            f = (cart[j] - lo) / mini_len                    # fractional in mini-cell
            site.fract = gemmi.Fractional(*f)
            site.occ = 1.0
            mini.add_site(site)

        dc = _make_density_calculator(scattering)
        dc.grid.spacegroup = sg_p1
        dc.grid.unit_cell = mini.cell
        dc.grid.set_size(G, G, G)
        dc.blur = blur
        if len(idx):
            str_mx = sx_to_mx_structure(mini, b_iso=b_iso)
            dc.put_model_density_on_grid(str_mx[0])

        full = np.array(dc.grid, dtype=np.float32)
        m = margin_vox
        return np.ascontiguousarray(full[m:m + block_vox, m:m + block_vox, m:m + block_vox])

    return build


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------
def tiled_patterson(structure: gemmi.SmallStructure,
                    q_max: float,
                    r_max: float,
                    *,
                    block_vox: Optional[int] = None,
                    margin_A: float = 3.0,
                    blur: float = 0.5,
                    b_iso: float = 0.0,
                    scattering: str = "xray",
                    apply_dv: bool = True,
                    cache_dir: Optional[str] = None,
                    disk_budget_bytes: int = 8 * 1024 ** 3,
                    mem_blocks: int = 4,
                    progress: bool = False) -> PattersonWindow:
    """Tiled 3D-Patterson on |r_α| ≤ r_max, no deblur, no window.

    Parameters
    ----------
    structure : gemmi.SmallStructure
        A single P1 configuration (orthogonal cell).
    q_max : float
        Data resolution (Å⁻¹). Sets the voxel Δr = π / q_max.
    r_max : float
        PDF half-range of interest (Å). Sets the window and, implicitly, the
        reciprocal bin — the two are a Fourier pair (see proposal).
    block_vox : int, optional
        Block side in voxels. Must be ≥ M = round(r_max/Δr) for the
        direct-neighbour-only decomposition to be valid; defaults to M.
    margin_A : float
        Guard band (Å) around each block when placing atoms, so Gaussian tails
        from neighbouring atoms are captured. Must exceed the density cutoff
        radius (a few Å for small blur).
    blur, b_iso, scattering :
        Passed through to the density build, exactly as in `sf_gemmi`.
        `scattering` selects the radiation ("xray", "neutron", or "electron").
    apply_dv : bool
        Multiply by dV² (dV = voxel volume) to match `sf_gemmi`'s scale.
    cache_dir, disk_budget_bytes, mem_blocks :
        Block-cache controls. `disk_budget_bytes ≤ 0` disables disk (evicted
        blocks are simply rebuilt from atoms).
    """
    cell = structure.cell
    _require_orthogonal(cell)

    dr = np.pi / q_max
    M = int(round(r_max / dr))
    if M < 1:
        raise ValueError("r_max is smaller than one voxel; increase r_max or q_max.")
    B = int(block_vox) if block_vox is not None else M
    if B < M:
        raise ValueError(f"block_vox ({B}) must be ≥ M ({M}) for direct-neighbour-only.")
    margin_vox = int(np.ceil(margin_A / dr))

    # Number of blocks per axis to cover the box (round the box up to n·B).
    lengths = np.array([cell.a, cell.b, cell.c], dtype=float)
    nvox = np.ceil(lengths / dr).astype(int)
    nb = np.ceil(nvox / B).astype(int)            # blocks per axis

    W = 2 * M + 1
    pw = np.zeros((W, W, W), dtype=np.float64)

    builder = _make_block_builder(structure, B, dr, margin_vox, blur, b_iso, scattering)
    owns_cache = cache_dir is None
    cdir = Path(cache_dir) if cache_dir is not None else Path(
        f".tiled_patterson_cache_{os.getpid()}")
    cache = BlockDensityCache(builder, cdir, disk_budget_bytes, mem_blocks)

    S = 2 * B                                     # linear-correlation FFT size (no wrap)
    n_blocks = int(np.prod(nb))
    try:
        for count, I in enumerate(np.ndindex(*nb)):
            if progress and count % max(1, n_blocks // 20) == 0:
                print(f"tiled_patterson: block {count + 1}/{n_blocks}")
            rho_I = cache.get(I)
            # forward transform of the outer block, reused across its neighbours
            FI = np.fft.rfftn(rho_I, s=(S, S, S))
            for D in _NEIGHBOUR_OFFSETS:
                J = (I[0] + D[0], I[1] + D[1], I[2] + D[2])
                if any(J[a] < 0 or J[a] >= nb[a] for a in range(3)):
                    continue
                rho_J = cache.get(J)
                FJ = np.fft.rfftn(rho_J, s=(S, S, S))
                # xc(t) = Σ_u ρ_I(u) ρ_J(u+t) = IFFT(conj(FI) · FJ)
                xc = np.fft.irfftn(np.conj(FI) * FJ, s=(S, S, S))
                xcc = np.fft.fftshift(xc)         # xcc[k] ↔ t = k - B
                _accumulate_pair(pw, xcc, D)
    finally:
        if owns_cache:
            cache.close()

    if apply_dv:
        dv = float(cell.volume) / float(np.prod(nvox))
        pw *= dv * dv

    return PattersonWindow(data=pw, r_step=dr, r_max=M * dr, blur=blur, cell=cell)


# ---------------------------------------------------------------------------
# reference (for validation only) — do NOT use on the full 4M-atom problem
# ---------------------------------------------------------------------------
def reference_patterson_from_density(density: NDArray[np.float64], M: int) -> NDArray[np.float64]:
    """Brute-force linear autocorrelation of a full density, cropped to |r|≤M.

    Only for tests: materialises the whole density and a zero-padded FFT, which
    is exactly what `tiled_patterson` avoids. Returns a (2M+1)³ window with the
    origin at the centre.
    """
    n = np.array(density.shape)
    s = tuple(2 * n)                                  # zero-pad → linear (no wrap)
    F = np.fft.rfftn(density, s=s)
    ac = np.fft.irfftn(np.abs(F) ** 2, s=s)           # ac[t], t ∈ [0, 2n) circular
    ac = np.fft.fftshift(ac)                           # centre at t = 0
    c = np.array(ac.shape) // 2
    sl = tuple(slice(c[a] - M, c[a] + M + 1) for a in range(3))
    return ac[sl]
