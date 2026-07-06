"""
Average diffuse scattering over a list of disordered configurations.

The hero function is `average_diffuse(structures, grid, ...)`. It takes a
Python list of realised P1 structures and returns a `DiffuseResult` containing:

- ⟨F⟩ — complex average structure factor (Bragg structure).
- ⟨I⟩ — averaged intensity (total scattering).
- ⟨I⟩ − |⟨F⟩|² — the diffuse map.
- Optionally, the per-snapshot complex SF cubes (opt-in; can be large).

Inputs to the structure list can be either `gemmi.SmallStructure` instances or
`DisorderedStructure` instances; the function converts the latter automatically.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union

import gemmi
import numpy as np
from numpy.typing import NDArray

from .grid import Grid
from .scattering import sf_gemmi
from .structures import DisorderedStructure


@dataclass
class DiffuseResult:
    """Container for the averaged maps over a collection of configurations."""

    grid: Grid
    n_configs: int
    average_F: NDArray[np.complex128]            # ⟨F⟩
    average_I: NDArray[np.float64]               # ⟨I⟩
    diffuse: NDArray[np.float64]                 # ⟨I⟩ − |⟨F⟩|²
    per_snapshot_F: Optional[List[NDArray[np.complex128]]] = None

    @property
    def average_bragg(self) -> NDArray[np.float64]:
        """|⟨F⟩|² — the average Bragg intensity (Friedel-symmetric, real)."""
        return np.abs(self.average_F) ** 2


def _to_small_structure(structure: Union[gemmi.SmallStructure, DisorderedStructure]) -> gemmi.SmallStructure:
    if isinstance(structure, DisorderedStructure):
        return structure.to_small_structure()
    return structure


def average_diffuse(structures: List[Union[gemmi.SmallStructure, DisorderedStructure]],
                    grid: Grid,
                    blur: float = 0.01,
                    crop: float = 2.0,
                    b_iso: float = 0.0,
                    scattering: str = "xray",
                    keep_per_snapshot: bool = False,
                    progress: bool = False) -> DiffuseResult:
    """Compute ⟨F⟩, ⟨I⟩, and ⟨I⟩−|⟨F⟩|² over a list of configurations.

    Parameters
    ----------
    structures: list of gemmi.SmallStructure or DisorderedStructure
        Realised P1 configurations to average over. All must share the same
        unit cell (the grid is fixed across the average).
    grid: Grid
        Reciprocal-space grid used for every snapshot.
    blur, crop, b_iso, scattering:
        Forwarded to `sf_gemmi` — see its docstring. `scattering` selects the
        radiation ("xray", "neutron", or "electron").
    keep_per_snapshot: bool
        If True, the returned result also carries every per-config SF cube
        in `result.per_snapshot_F`. Off by default for memory reasons.
    progress: bool
        If True, print "calculating config i/N" lines as we iterate.

    Returns
    -------
    DiffuseResult
    """
    if not structures:
        raise ValueError("`structures` must contain at least one configuration.")

    n = len(structures)
    accum_F = None
    accum_I = None
    per_snapshot: Optional[List[NDArray[np.complex128]]] = [] if keep_per_snapshot else None

    for i, raw in enumerate(structures):
        if progress:
            print(f"calculating config {i + 1}/{n}")

        struct = _to_small_structure(raw)
        sf = sf_gemmi(struct, grid, blur=blur, crop=crop, b_iso=b_iso,
                      scattering=scattering)

        if accum_F is None:
            accum_F = np.zeros_like(sf)
            accum_I = np.zeros(sf.shape, dtype=np.float64)

        accum_F += sf
        accum_I += np.abs(sf) ** 2

        if per_snapshot is not None:
            per_snapshot.append(sf)

    average_F = accum_F / n
    average_I = accum_I / n
    diffuse = average_I - np.abs(average_F) ** 2

    return DiffuseResult(
        grid=grid,
        n_configs=n,
        average_F=average_F,
        average_I=average_I,
        diffuse=diffuse,
        per_snapshot_F=per_snapshot,
    )
