# Possible bug in `gemmi::atom_to_site` (`interop.hpp`)

Tested with **gemmi 0.7.3** on macOS, Python 3.13.

## Summary

`gemmi::atom_to_site` (and therefore `mx_to_sx_structure`) skips the
U_cart → U_cif basis change whenever **any one** cell angle equals 90°.
That branch produces a `SmallStructure` whose `site.aniso` field does
not satisfy the IUCr convention that the rest of gemmi (e.g.
`StructureFactorCalculatorX::calculate_sf_from_small_structure` and
the CIF reader) uses.

Net effect: `mx_to_sx_structure(st)` followed by
`calculate_sf_from_small_structure(...)` returns structure factors that
do **not** agree with `calculate_sf_from_model(...)` on the same
structure for hexagonal / trigonal / monoclinic cells. The disagreement
grows quickly with `|U|` and `|hkl|` and is order-of-tens-of-percent for
realistic ADP magnitudes.

## The code

`gemmi/include/gemmi/interop.hpp`, lines 26–39:

```cpp
if (atom.aniso.nonzero()) {
  if (cell.alpha == 90. || cell.beta == 90. || cell.gamma == 90.) {
    site.aniso = atom.aniso.scaled(1.0);          // identity copy
  } else {
    SMat33<double> t = atom.aniso.transformed_by<>(cell.frac.mat);
    Vec3 v = {1.0 / cell.ar, 1.0 / cell.br, 1.0 / cell.cr};
    site.aniso = {t.u11 * v.x * v.x,
                  t.u22 * v.y * v.y,
                  t.u33 * v.z * v.z,
                  t.u12 * v.x * v.y,
                  t.u13 * v.x * v.z,
                  t.u23 * v.y * v.z};
  }
}
```

The first branch is taken whenever **any** of α, β, γ equals 90°. That
includes hexagonal/trigonal (α=β=90, γ=120) and monoclinic
(α=γ=90, β≠90). Mathematically, `U_cart == U_cif` only when **all
three** angles are 90° (fully orthogonal). The condition should be
`&&`, not `||`.

## What gemmi does *elsewhere* with `site.aniso`

The rest of gemmi treats `site.aniso` as the standard IUCr CIF
`U_aniso` (the "fractional" basis), i.e. the Debye–Waller factor is

```
DW(hkl) = exp(−2π² (U_11 h² a*² + U_22 k² b*² + U_33 l² c*²
                    + 2 U_12 h k a* b*
                    + 2 U_13 h l a* c*
                    + 2 U_23 k l b* c*))
```

This was verified by:

1. Writing a CIF with hexagonal cell (7,7,6,90,90,120) and an
   `_atom_site_aniso_U_*` block.
2. Reading it with `gemmi.read_small_structure` — gemmi stores the
   CIF values verbatim in `site.aniso`.
3. Calling `calculate_sf_from_small_structure` and comparing with the
   value computed by hand using the IUCr DW formula above.
4. The two agree to four decimal places across every reflection
   tested.

So `calculate_sf_from_small_structure` and `read_small_structure`
agree with the IUCr convention; only `atom_to_site` / `mx_to_sx_structure`
disagree.

## Reproducer 1 — atom_to_site copies U_cart verbatim instead of converting

```python
import gemmi, numpy as np

cell = gemmi.UnitCell(7.0, 7.0, 6.0, 90.0, 90.0, 120.0)

# Build a 1-atom mx structure with a non-trivial U_cart.
st = gemmi.Structure(); st.cell = cell; st.spacegroup_hm = "P 1"
m = gemmi.Model("1"); c = gemmi.Chain("A"); r = gemmi.Residue()
at = gemmi.Atom(); at.name = "C0"; at.element = gemmi.Element("C")
at.pos = cell.orthogonalize(gemmi.Fractional(0.25, 0.25, 0.25))
at.b_iso = 0.0; at.occ = 1.0
at.aniso = gemmi.SMat33f(0.012, 0.013, 0.020, 0.0021, 0.0007, 0.0003)
r.add_atom(at); c.add_residue(r); m.add_chain(c); st.add_model(m)

sx = gemmi.mx_to_sx_structure(st)
print("site.aniso (gemmi)  :", list(sx.sites[0].aniso))

# Canonical mx -> sx formula (see interop.hpp 'else' branch):
F = np.array(cell.frac.mat)
U = np.array([[at.aniso.u11, at.aniso.u12, at.aniso.u13],
              [at.aniso.u12, at.aniso.u22, at.aniso.u23],
              [at.aniso.u13, at.aniso.u23, at.aniso.u33]])
t = F @ U @ F.T
rc = cell.reciprocal()
v = np.array([1.0/rc.a, 1.0/rc.b, 1.0/rc.c])
U_cif = t * np.outer(v, v)
print("U_cif (canonical)   :",
      [U_cif[0,0], U_cif[1,1], U_cif[2,2], U_cif[0,1], U_cif[0,2], U_cif[1,2]])
```

Output:

```
site.aniso (gemmi)  : [0.012, 0.013, 0.020, 0.0021, 0.0007, 0.0003]
U_cif (canonical)   : [0.01406865, 0.013, 0.020, 0.00831865, 0.00075622, 0.0003]
```

`U_11` is off by 17%, `U_12` by a factor of ~4. (The disagreement is
exactly on the components affected by the in-plane γ ≠ 90.)

## Reproducer 2 — observable as F(hkl) disagreement

Picking up the same `st` from Reproducer 1 but with larger U
(0.12, 0.13, 0.20, 0.04, 0.02, 0.01) to make the effect obvious:

```python
sx_via = gemmi.mx_to_sx_structure(st)
calc = gemmi.StructureFactorCalculatorX(cell)

for hkl in [(1,0,0),(2,0,0),(3,0,0),(1,1,0),(2,1,0),(0,0,1),(0,0,2),(3,2,1)]:
    f_sx = calc.calculate_sf_from_small_structure(sx_via, list(hkl))
    f_mx = calc.calculate_sf_from_model(st[0],          list(hkl))
    print(f"{str(hkl):<10}  sx {abs(f_sx):8.3f}   mx {abs(f_mx):8.3f}   "
          f"ratio {abs(f_sx)/abs(f_mx):.3f}")
```

Output:

```
(1, 0, 0)   sx   58.894   mx   57.730   ratio 1.020
(2, 0, 0)   sx   44.363   mx   40.960   ratio 1.083
(3, 0, 0)   sx   29.011   mx   24.243   ratio 1.197
(1, 1, 0)   sx   49.309   mx   45.336   ratio 1.088
(2, 1, 0)   sx   35.591   mx   28.909   ratio 1.231
(0, 0, 1)   sx   56.243   mx   56.243   ratio 1.000
(0, 0, 2)   sx   36.951   mx   36.951   ratio 1.000
(3, 2, 1)   sx   11.807   mx    6.667   ratio 1.771
```

Reflections lying along the c-axis (`0,0,l`) are unaffected because
`U_33` and `c*` aren't touched by the γ rotation. Everything in the
ab plane is wrong; the worst case shown is 77% above the correct
amplitude.

## Suggested fix

```diff
- if (cell.alpha == 90. || cell.beta == 90. || cell.gamma == 90.) {
+ if (cell.alpha == 90. && cell.beta == 90. && cell.gamma == 90.) {
      site.aniso = atom.aniso.scaled(1.0);
  } else {
      SMat33<double> t = atom.aniso.transformed_by<>(cell.frac.mat);
      ...
  }
```

## Scope of impact

- **Hexagonal, trigonal, rhombohedral** cells (γ = 60° or 120°, α = β = 90°).
- **Monoclinic** cells (β ≠ 90°, α = γ = 90°).
- Any cell where exactly one or two of {α, β, γ} equal 90°.

Cells unaffected:

- **Triclinic** — none of α, β, γ equals 90°, so the correct branch is taken.
- **Cubic / tetragonal / orthorhombic** — all three angles equal 90°,
  and the identity copy is correct.

## Note

`atom_to_site` is also called from `mx_to_sx_structure`, so any user
pipeline that does `Structure → SmallStructure → calculate_sf` is
affected. The CIF-reading path
(`read_small_structure → calculate_sf_from_small_structure`) is **not**
affected — gemmi stores the CIF's IUCr values verbatim and computes
with the IUCr formula correctly.
