# Style notes & grudges to discuss later

Things I noticed while consolidating the three CalculateScattering.py copies
into this package. None are blockers; collected here to discuss in a single
pass rather than peppering each file with FIXMEs.

## Substantive (functional)

1. **Missing FFT normalization in the original `sf_gemmi`.**
   The Dy467 / Seminar 7 code does
   `sf_grid = np.fft.fftn(np.array(dencalc.grid))` with no factor.
   The crystallographic structure-factor convention requires multiplying by
   `dV = V / N_padded` so that `F(000) = ∑ Zⱼ`. I added this. The diffuse
   pipeline downstream only compares relative intensities, so the omission
   was invisible — but it means *every* SF cube saved to Yell so far has been
   off by a constant scale factor.
   - **Action**: confirm with you that this is fine, and decide whether to
     publish a one-line fix to the Dy467 script.

2. **Deblur factor.** The empirical
   `exp(blur · 0.25 · q² · 4)` simplifies to `exp(blur · stol²)`, which is the
   principled form. I rewrote it as `exp(blur · stol²)` literally, and removed
   the `× 4` comment annotations. With this and the dV fix above, the fast
   path matches the direct path at the per-percent level on resolved grids.
   - **Action**: confirm the regression you saw at `blur=0.1` is gone with
     the dV fix in place. If not, we'll need to dig further.

3. **`b_iso` choice in `sx_to_mx_structure`.** Dy467 used 0; Seminar 7 used
   0.5 (as a numerical stabiliser at low blur). Made it a kwarg with default
   0 — call sites that care can pass it. Probably worth a brief note in your
   teaching notebook so the value isn't a magic number.

4. **`save_to_yell(atoms, data)` in the originals writes `np.real(F)`** despite
   the function name and the comment about diffuse scattering. I did not port
   that function; the new `save_to_yell(result, ...)` writes the three
   averaged maps (diffuse, |⟨F⟩|², ⟨I⟩), all real, via `save2yellS`.

5. **`calculate_sf` (pure-numpy direct) and `prepare_atoms`** were skipped.
   `prepare_atoms` hard-coded occupancy = 1 and ADP = 0. `calculate_sf` built
   `q_vectors` in r.l.u. (h,k,l) but computed `metric_matrix = cell @ cell.T`
   in real space — the convention mixing produces a wrong `q²` for
   non-orthogonal cells (and even for orthogonal ones the formula isn't right
   for what's labelled). Easier to delete than to repair, especially because
   `sf_gemmi_direct` already does the slow ground-truth job correctly.

6. **`prepare_atoms` discarded fractional coordinates** for atoms living
   outside `[0, 1)`. The function did `atom_data[i]['pos'] = atom.scaled_position`
   which ASE returns *unwrapped* for some inputs. The new path goes through
   `DisorderedStructure.to_small_structure` which is explicit about the
   "x in [0, supercell_size)" convention and divides by the supercell — no
   silent wrapping.

## Cosmetic (Python style)

7. **Duplicated imports** in the originals:
   `from typing import Dict, NamedTuple, Tuple` on line 1 and
   `from typing import List, Tuple, Union` on line 7. Merged into a single
   import per file.

8. **`def padding(self, crop : int) -> "Grid"`** — return annotation says
   `Grid` but the function returns an `NDArray[int]`. Fixed the annotation to
   match reality.

9. **`crop: int`** for what is clearly a float (1.8 in defaults). Changed to
   `float`.

10. **`save2yellS` reassigns `supercell` from a tuple to a numpy array** in
    its body. Survivable but confusing if you ever try to use it after the
    call. Left as-is to keep the function byte-identical to the existing
    one used in Dy467.

11. **`StructureFactors` returns the underlying `Grid`**, which is mutable.
    If the user happens to mutate `grid.no_pixels` after the fact, the
    invariant breaks. Low priority; would be fixed by making `Grid` frozen
    or having `StructureFactors` deep-copy on construction. @AS COMMENT: Make Grid frozen, init-only const.

12. **`gemmi.Atom().name = f"{site.element.name}{i}"`** in
    `sx_to_mx_structure` — atom names aren't unique across elements (e.g.
    `Dy0`, `Dy1`, … `O500` etc are fine, but `Dy0` and `Si0` could collide
    if you accidentally renumbered). gemmi tolerates it but it's a footgun.
    Not changed.

13. The Seminar-7 `CrystalStructure.calculate_scattering` does:
    `write(temp_path, atoms, format='cif'); structure = gemmi.read_small_structure(temp_path)`.
    Replaced with an in-memory `DisorderedStructure.to_small_structure()`.
    Cleaner; ASE no longer needed in the hot path.

## Things I deliberately did not change yet

- **`save2yellS` is byte-identical** to the original. Don't want to introduce
  format drift before checking with you whether anything reads these files
  in a way that's sensitive to e.g. dtype. @AS coment: Don't stick to format, be reasonable editing it to make it better.
- **No multi-model file support.** A `gemmi.read_small_structure` reads a
  single configuration, which is what every caller expects. @AS: assume only one configuration per cif, add this into README.MD description.
- **No symmetry expansion.** Package assumes P1 input — same as before.
