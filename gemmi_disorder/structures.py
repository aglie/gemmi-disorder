"""
DisorderedStructure — lightweight container for a P1 atomic configuration.

This is the same idea as the `CrystalStructure` dataclass in the Seminar 7
copy of CalculateScattering.py, except the conversion to a gemmi structure
does NOT go through a temporary CIF file. We build a `gemmi.SmallStructure`
in memory directly, which is faster and keeps unit-cell precision intact.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import gemmi


@dataclass
class DisorderedStructure:
    """
    A single realised disordered atomic configuration.

    Parameters
    ----------
    cell_parameters: Tuple[float, float, float, float, float, float]
        Unit cell parameters (a, b, c, alpha, beta, gamma) of the UNDERLYING
        unit cell — i.e. before supercell expansion. Lengths in Å, angles
        in degrees.
    atoms: List[Tuple[str, float, float, float]]
        Atoms as (element_symbol, x, y, z). Coordinates are fractional with
        respect to the supercell: x in [0, nx), y in [0, ny), z in [0, nz).
        This matches the Seminar 7 convention.
    supercell: Tuple[int, int, int]
        Supercell size (nx, ny, nz). The supercell unit cell sides are
        (a·nx, b·ny, c·nz, α, β, γ).
    name: str
        Optional structure name (passed to gemmi).
    """

    cell_parameters: Tuple[float, float, float, float, float, float]
    atoms: List[Tuple[str, float, float, float]]
    supercell: Tuple[int, int, int] = (1, 1, 1)
    name: str = "disordered"

    def supercell_cell(self) -> gemmi.UnitCell:
        """Return the expanded supercell as a gemmi.UnitCell."""
        a, b, c, alpha, beta, gamma = self.cell_parameters
        nx, ny, nz = self.supercell
        return gemmi.UnitCell(a * nx, b * ny, c * nz, alpha, beta, gamma)

    def to_small_structure(self) -> gemmi.SmallStructure:
        """
        Build a `gemmi.SmallStructure` in P1 from this configuration.

        Atom positions are mapped to fractional coordinates of the supercell
        unit cell (divide by `supercell`).
        """
        nx, ny, nz = self.supercell
        out = gemmi.SmallStructure()
        out.name = self.name
        out.cell = self.supercell_cell()
        out.spacegroup_hm = "P 1"

        for i, (element, x, y, z) in enumerate(self.atoms):
            site = gemmi.SmallStructure.Site()
            site.label = f"{element}{i}"
            site.type_symbol = element
            site.element = gemmi.Element(element)
            site.fract = gemmi.Fractional(x / nx, y / ny, z / nz)
            site.occ = 1.0
            # site.aniso is left at zero (the default). Non-zero ADPs are
            # rejected later in scattering.sf_gemmi via _check_zero_aniso.
            out.add_site(site)

        return out


# Convenience: build a DisorderedStructure from numpy arrays. Kept very thin
# so the dataclass stays the canonical representation.
def disordered_structure_from_arrays(elements: List[str],
                                     positions,
                                     cell_parameters: Tuple[float, float, float, float, float, float],
                                     supercell: Tuple[int, int, int] = (1, 1, 1),
                                     name: str = "disordered") -> DisorderedStructure:
    """Build a DisorderedStructure from (elements, positions) arrays.

    Parameters
    ----------
    elements: list[str] of length N_atoms
    positions: array-like of shape (N_atoms, 3), fractional w.r.t. supercell.
    """
    atoms = [(elements[i], float(positions[i][0]), float(positions[i][1]), float(positions[i][2]))
             for i in range(len(elements))]
    return DisorderedStructure(
        cell_parameters=cell_parameters,
        atoms=atoms,
        supercell=supercell,
        name=name,
    )
