"""
I/O helpers — saving averaged maps in Yell 1.0 HDF5 format.

`save2yellS` is verbatim from the existing CalculateScattering.py used in the
Dy467 production pipeline. `save_to_yell` is a thin wrapper that, given a
`DiffuseResult`, writes three .h5 files (diffuse / averaged / total) using
the supercell convention.
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

    Parameters
    ----------
    output_filename: str | Path
        Path to the output .h5 file.
    intensity: array
        3D real array of intensity values (shape = grid pixels).
    cell: gemmi.UnitCell
        Underlying unit cell (NOT the supercell — the supercell expansion is
        applied here from `supercell`).
    supercell: (nx, ny, nz)
        Supercell size used when computing `intensity`.
    """
    unit_cell = [cell.a / supercell[0], cell.b / supercell[1], cell.c / supercell[2],
                 cell.alpha, cell.beta, cell.gamma]
    supercell = np.array(supercell)
    # intensity = fftshift(intensity)
    hklmax = np.array(intensity.shape) / 2

    output = h5py.File(output_filename, 'w')

    output['data'] = intensity
    output['format'] = b'Yell 1.0'  # formatting string
    output['is_direct'] = False  # whether the data is in real or reciprocal space. Scattering data is in reciprocal space
    output['lower_limits'] = -hklmax / supercell  # the smallest hkl index for this dataset
    output['step_sizes'] = 1 / supercell
    output['unit_cell'] = unit_cell

    output.close()


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
        Underlying unit cell. (We don't store this in `DiffuseResult` — passing
        it explicitly keeps the result object purely numerical.)
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
