"""
Run a short ASE molecular-dynamics trajectory of FCC Cu, sample snapshots,
and feed them to gemmi-disorder's `average_diffuse`.

The point of this script is to (a) exercise the package on a real ensemble of
disordered atomic configurations rather than hand-crafted toys, and (b) sanity
check the output:

- |<F>|^2 should look like a perfect Bragg pattern at integer hkl (the
  time-averaged structure is FCC, sharp peaks at allowed reflections).
- <I> - |<F>|^2 should be a smooth diffuse background that grows with q
  (thermal diffuse scattering).
- Sharp Bragg peaks should not bleed into the diffuse map.

Usage:
    python examples/md_fcc_diffuse.py
"""

from __future__ import annotations

import time

import numpy as np
from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from gemmi_disorder import (
    DisorderedStructure,
    Grid,
    average_diffuse,
)


def main():
    # ----- 1. Build the supercell ------------------------------------------------
    # Cu, conventional cubic cell (4 atoms), 6x6x6 supercell -> 864 atoms.
    a0 = 3.61  # Å, equilibrium FCC Cu
    unit = bulk("Cu", "fcc", a=a0, cubic=True)
    supercell = (6, 6, 6)
    atoms = unit.repeat(supercell)
    atoms.calc = EMT()
    print(f"system: {len(atoms)} atoms, "
          f"supercell {supercell}, conventional cell a={a0} Å")

    # ----- 2. MD setup -----------------------------------------------------------
    T = 600.0  # K
    MaxwellBoltzmannDistribution(atoms, temperature_K=T)

    dyn = Langevin(
        atoms,
        timestep=2.0 * units.fs,
        temperature_K=T,
        friction=0.02,  # 1/fs
    )

    # Equilibrate
    print("equilibrating ...")
    t0 = time.time()
    dyn.run(500)  # 1 ps
    print(f"  equilibration done in {time.time() - t0:.1f}s, "
          f"final T = {atoms.get_temperature():.1f} K")

    # ----- 3. Collect snapshots --------------------------------------------------
    n_snapshots = 20
    stride = 100  # 0.2 ps between snaps
    snapshots: list[DisorderedStructure] = []

    print(f"collecting {n_snapshots} snapshots, {stride} steps apart ...")
    t0 = time.time()
    for i in range(n_snapshots):
        dyn.run(stride)

        # ASE scaled_positions are wrt the supercell box. We want them in
        # supercell-fractional coordinates (i.e. each in [0, nx)) so that
        # DisorderedStructure can do the supercell-to-cell rescaling.
        scaled = atoms.get_scaled_positions(wrap=True)
        nx, ny, nz = supercell
        scaled_in_supercell = scaled * np.array([nx, ny, nz])

        atom_tuples = [
            ("Cu", float(p[0]), float(p[1]), float(p[2]))
            for p in scaled_in_supercell
        ]
        snapshots.append(DisorderedStructure(
            cell_parameters=(a0, a0, a0, 90.0, 90.0, 90.0),
            atoms=atom_tuples,
            supercell=supercell,
            name=f"snap{i:02d}",
        ))
    print(f"  {n_snapshots} snapshots collected in {time.time() - t0:.1f}s, "
          f"T = {atoms.get_temperature():.1f} K")

    # ----- 4. Average diffuse scattering -----------------------------------------
    # hkl in r.l.u. relative to the underlying (conventional) cell. Step
    # 1/supercell, half-range hkl_max = 3 -> covers ±3 r.l.u. in each axis.
    grid = Grid.from_supercell(supercell=supercell, hkl_max=3)
    print(f"grid: pixels {grid.no_pixels.tolist()}, "
          f"step {grid.step_sizes.tolist()}, "
          f"lower {grid.lower_limits.tolist()}")

    t0 = time.time()
    result = average_diffuse(
        snapshots,
        grid,
        blur=0.5,     # small extra B for FFT stability
        crop=2.0,
        b_iso=0.0,    # no extra ADP — disorder comes from the MD displacements
        progress=True,
    )
    print(f"average_diffuse done in {time.time() - t0:.1f}s")

    # ----- 5. Inspect ------------------------------------------------------------
    bragg = result.average_bragg  # |<F>|^2
    total = result.average_I
    diffuse = result.diffuse

    # Centre of grid corresponds to hkl = (0, 0, 0).
    centre = tuple(g // 2 for g in grid.no_pixels)
    z_total = sum(s.element.atomic_number
                  for ds in snapshots[:1]
                  for s in ds.to_small_structure().sites)
    print()
    print(f"F(000)   = {result.average_F[centre]:.2f}  "
          f"(expected sum(Z) = {z_total})")
    print(f"|<F>|^2(000) = {bragg[centre]:.1f}")
    print(f"<I>(000)     = {total[centre]:.1f}")
    print(f"diffuse(000) = {diffuse[centre]:.4f}  "
          f"(should be ~0 — Bragg peak is all coherent)")
    print()
    print(f"max Bragg  = {bragg.max():.2e}")
    print(f"max diffuse = {diffuse.max():.2e}")
    print(f"diffuse / Bragg ratio (max-to-max) = "
          f"{diffuse.max() / bragg.max():.2e}")
    # Sanity: diffuse should be non-negative everywhere (modulo num noise).
    assert diffuse.min() > -1e-3 * bragg.max(), \
        f"diffuse min = {diffuse.min()}, suspiciously negative"
    print(f"diffuse min  = {diffuse.min():.2e}  (should be ~0 from below)")

    # Optional: a few example (hkl) extractions. Allowed FCC reflections
    # (h,k,l all even or all odd) should dominate <F>; others should be
    # suppressed.
    def index_of(hkl):
        return tuple(int(round((hkl[i] - grid.lower_limits[i]) / grid.step_sizes[i]))
                     for i in range(3))

    print()
    print(f"{'hkl':<10} {'|<F>|^2':>12} {'<I>':>12} {'diffuse':>12}  rule")
    examples = [
        ((0, 0, 0), "F(000)"),
        ((1, 1, 1), "FCC allowed (all odd)"),
        ((2, 0, 0), "FCC allowed (all even)"),
        ((2, 2, 0), "FCC allowed (all even)"),
        ((2, 2, 2), "FCC allowed (all even)"),
        ((1, 0, 0), "FCC FORBIDDEN (mixed)"),
        ((1, 1, 0), "FCC FORBIDDEN (mixed)"),
        ((2, 1, 0), "FCC FORBIDDEN (mixed)"),
    ]
    for hkl, label in examples:
        idx = index_of(hkl)
        print(f"{str(hkl):<10} {bragg[idx]:>12.2f} {total[idx]:>12.2f} "
              f"{diffuse[idx]:>12.2f}  {label}")

    # Save the diffuse map for inspection (npz, easy to load in a notebook).
    out = "md_fcc_diffuse.npz"
    np.savez(
        out,
        average_F=result.average_F,
        average_I=result.average_I,
        diffuse=result.diffuse,
        grid_lower=grid.lower_limits,
        grid_step=grid.step_sizes,
        grid_npix=grid.no_pixels,
    )
    print(f"\nsaved arrays to {out}")


if __name__ == "__main__":
    main()
