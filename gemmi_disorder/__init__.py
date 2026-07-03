"""
gemmi-disorder — diffuse scattering from disordered atomic configurations.

Thin wrapper around gemmi that computes diffuse scattering from a collection
of realised P1 configurations using the density-on-grid + FFT fast path.

Quick start
-----------
>>> import gemmi
>>> from gemmi_disorder import Grid, average_diffuse, save_to_yell
>>>
>>> structures = [gemmi.read_small_structure(f) for f in cif_paths]
>>> grid = Grid.from_supercell(supercell=(6, 6, 12), hkl_max=14)
>>> result = average_diffuse(structures, grid, blur=0.01)
>>> save_to_yell(result, cell=structures[0].cell, supercell=(6, 6, 12))
"""

from .grid import Grid, StructureFactors
from .structures import DisorderedStructure, disordered_structure_from_arrays
from .scattering import (
    sf_gemmi,
    sf_gemmi_direct,
    sx_to_mx_structure,
    sx_aniso_to_cart,
    get_form_factors,
    generate_q_vectors,
    calculate_stol_squared,
)
from .averaging import average_diffuse, DiffuseResult
from .io import save2yellS, save_to_yell
from .pdf import tiled_patterson, PattersonWindow, BlockDensityCache

__all__ = [
    "Grid",
    "StructureFactors",
    "DisorderedStructure",
    "disordered_structure_from_arrays",
    "sf_gemmi",
    "sf_gemmi_direct",
    "sx_to_mx_structure",
    "sx_aniso_to_cart",
    "get_form_factors",
    "generate_q_vectors",
    "calculate_stol_squared",
    "average_diffuse",
    "DiffuseResult",
    "save2yellS",
    "save_to_yell",
    "tiled_patterson",
    "PattersonWindow",
    "BlockDensityCache",
]
