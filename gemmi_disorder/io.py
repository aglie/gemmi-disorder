"""
I/O helpers — saving averaged maps in Yell 1.0 HDF5 format.

Yell 1.0 file layout (all entries top-level datasets):

    /data           3D float array of intensity values
    /format         b'Yell 1.0'
    /is_direct      False (this package writes reciprocal-space data only)
    /lower_limits   3-tuple, hkl of voxel data[0,0,0] in r.l.u.
    /step_sizes     3-tuple, step in r.l.u. per voxel along each axis
    /unit_cell      6-tuple, (a, b, c, alpha, beta, gamma) of the underlying
                    motif unit cell (i.e. supercell divided by `supercell`)
"""

from pathlib import Path
from typing import Tuple, Union

import gemmi
import h5py
import numpy as np
from numpy.typing import NDArray

from .averaging import DiffuseResult


def save2yellS(output_filename: Union[str, Path],
               intensity: NDArray,
               cell: gemmi.UnitCell,
               supercell: Tuple[int, int, int]) -> None:
    """Save a 3D intensity array to a Yell 1.0 HDF5 file.

    The grid is assumed to be centred on h=k=l=0 — `lower_limits` is set to
    `-(shape / 2) / supercell` so that voxel `data[shape/2]` corresponds to
    the origin of reciprocal space.

    Parameters
    ----------
    output_filename: str | Path
        Path to the output .h5 file.
    intensity: array
        3D real array of intensity values (shape = grid pixels).
    cell: gemmi.UnitCell
        Underlying *motif* unit cell, before supercell expansion. The cell
        written to file is divided by `supercell`.
    supercell: (nx, ny, nz)
        Supercell size that was used to compute `intensity`.
    """
    nx, ny, nz = supercell
    motif_cell = [cell.a / nx, cell.b / ny, cell.c / nz,
                  cell.alpha, cell.beta, cell.gamma]
    sc = np.asarray(supercell, dtype=float)
    hkl_half = np.asarray(intensity.shape, dtype=float) / 2.0

    with h5py.File(output_filename, "w") as out:
        out["data"] = intensity
        out["format"] = b"Yell 1.0"
        out["is_direct"] = False
        out["lower_limits"] = -hkl_half / sc
        out["step_sizes"] = 1.0 / sc
        out["unit_cell"] = motif_cell


def save_to_yell(result: DiffuseResult,
                 cell: gemmi.UnitCell,
                 supercell: Tuple[int, int, int],
                 prefix: Union[str, Path] = ".") -> None:
    """Write the three averaged intensity maps from a `DiffuseResult` to Yell .h5 files.

    Files written (under `prefix`):
        - diffuse_intensity.h5   ⟨I⟩ − |⟨F⟩|²
        - av_intensity.h5        |⟨F⟩|²  (the average Bragg structure)
        - tot_intensity.h5       ⟨I⟩

    Parameters
    ----------
    result: DiffuseResult
        Output of `average_diffuse`.
    cell: gemmi.UnitCell
        Underlying motif unit cell (NOT the supercell). The file's `unit_cell`
        entry will be this cell divided by `supercell` — kept consistent with
        the Dy467 / Yell convention.
    supercell: (nx, ny, nz)
        Supercell size used for the calculation.
    prefix: str | Path
        Directory in which to write the three files. Created if needed.
    """
    prefix = Path(prefix)
    prefix.mkdir(parents=True, exist_ok=True)

    save2yellS(prefix / "diffuse_intensity.h5", result.diffuse, cell, supercell)
    save2yellS(prefix / "av_intensity.h5", result.average_bragg, cell, supercell)
    save2yellS(prefix / "tot_intensity.h5", result.average_I, cell, supercell)
