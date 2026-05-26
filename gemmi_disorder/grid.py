"""
Grid and StructureFactors containers.

The `Grid` dataclass and `StructureFactors` NamedTuple are taken verbatim from
the existing CalculateScattering.py used in the Dy467 and Seminar 7 work — the
only addition is a `Grid.from_supercell` factory that matches the diffuse
scattering mental model (1 r.l.u. step = 1 supercell vector).
"""

from dataclasses import dataclass
from typing import List, NamedTuple, Tuple, Union

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Grid:
    """
    Represents a 3D calculation grid for structure factors.

    Immutable: the three arrays are validated and converted to read-only
    numpy arrays in `__post_init__`. To make a modified grid, construct a
    new one (or use the `Grid.from_supercell` factory).

    Parameters:
    -----------
    lower_limits: NDArray[np.float64]
        Starting coordinates for the grid [x_min, y_min, z_min]
    step_sizes: NDArray[np.float64]
        Step size in each direction [dx, dy, dz]
    no_pixels: NDArray[np.int64]
        Number of grid points in each direction [nx, ny, nz]
    """
    lower_limits: Union[List[float], NDArray[np.float64]]
    step_sizes: Union[List[float], NDArray[np.float64]]
    no_pixels: Union[List[int], NDArray[np.int64]]

    def __post_init__(self):
        # Validate input lengths
        if not all(len(x) == 3 for x in [self.lower_limits, self.step_sizes, self.no_pixels]):
            raise ValueError("All grid parameters must be 3D (length 3)")

        # Convert lists to numpy arrays and mark them read-only. The
        # dataclass is frozen, so attribute assignment goes via
        # object.__setattr__.
        ll = np.array(self.lower_limits, dtype=float)
        ss = np.array(self.step_sizes, dtype=float)
        np_ = np.array(self.no_pixels, dtype=int)
        for arr in (ll, ss, np_):
            arr.flags.writeable = False
        object.__setattr__(self, "lower_limits", ll)
        object.__setattr__(self, "step_sizes", ss)
        object.__setattr__(self, "no_pixels", np_)

    @classmethod
    def from_supercell(cls,
                       supercell: Tuple[int, int, int],
                       hkl_max: Union[int, Tuple[int, int, int]]) -> "Grid":
        """
        Construct a Grid using the supercell convention used in diffuse scattering.

        The grid covers integer Miller indices in [-hkl_max, +hkl_max] sampled at
        intervals of 1/supercell along each axis — i.e. one grid step per
        supercell vector. The returned grid is centred on h=k=l=0.

        Parameters
        ----------
        supercell: (nx, ny, nz)
            Supercell size along a, b, c.
        hkl_max: int or (hmax, kmax, lmax)
            Half-extent of the grid in each direction (in r.l.u.).

        Returns
        -------
        Grid with no_pixels = 2 * hkl_max * supercell and step_sizes = 1/supercell.
        """
        nx, ny, nz = supercell
        if isinstance(hkl_max, int):
            hmax, kmax, lmax = hkl_max, hkl_max, hkl_max
        else:
            hmax, kmax, lmax = hkl_max

        step_sizes = [1.0 / nx, 1.0 / ny, 1.0 / nz]
        no_pixels = [2 * hmax * nx, 2 * kmax * ny, 2 * lmax * nz]
        lower_limits = [-hmax, -kmax, -lmax]

        return cls(
            lower_limits=lower_limits,
            step_sizes=step_sizes,
            no_pixels=no_pixels,
        )

    def reciprocal_grid(self) -> 'Grid':
        """
        Returns corresponding grid in reciprocal space
        """
        # Calculate reciprocal grid steps and limits
        rec_step_sizes = np.zeros(3)
        rec_lower_limits = np.zeros(3)

        for i in range(3):
            if abs(self.lower_limits[i]) != 0:
                rec_step_sizes[i] = -0.5 / self.lower_limits[i]
                rec_lower_limits[i] = -0.5 / self.step_sizes[i]

        return Grid(
            lower_limits=rec_lower_limits,
            step_sizes=rec_step_sizes,
            no_pixels=self.no_pixels
        )

    def padding(self, crop: float) -> NDArray[np.int64]:
        padded_size = np.array(np.ceil(self.no_pixels * crop / 2) * 2, dtype=int)
        padding = np.array(np.round((padded_size - self.no_pixels) / 2), dtype=int)
        return padding

    def pad(self, crop: float) -> "Grid":
        padded_size = np.array(np.ceil(self.no_pixels * crop / 2) * 2, dtype=int)
        padding = np.array(np.round((self.no_pixels - padded_size) / 2), dtype=int)
        new_ll = self.lower_limits - padding * self.step_sizes
        return Grid(lower_limits=new_ll,
                    step_sizes=self.step_sizes,
                    no_pixels=padded_size)


class StructureFactors(NamedTuple):
    """Holds structure factor calculation results"""
    values: NDArray[np.complex64]  # Complex array of structure factors
    grid: Grid  # Original calculation grid

    def get_phases(self) -> NDArray[np.float64]:
        """Return phases in degrees"""
        return np.angle(self.values, deg=True)

    def get_amplitudes(self) -> NDArray[np.float64]:
        """Return amplitudes"""
        return np.abs(self.values)

    def get_intensities(self) -> NDArray[np.float64]:
        """Return intensities (amplitude squared)"""
        return np.abs(self.values) ** 2
