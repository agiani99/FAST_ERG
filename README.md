# FAST_ERG

Fast, topological **ErG-style pharmacophore fingerprints** (2D, no conformers) with a C++ kernel and an
**exact similarity search** over sharded indexes of 100M+ molecules.

* **ErG v2 fingerprint** – 321 values per molecule: 6 pharmacophore-site counts + 21 site-pair types × 15 topological-distance bins.
* **C++ / OpenMP kernel** (RDKit + pybind11) – ~5.5× faster than the Python reference on one core, ~38× on all cores, bit-identical output.
* **Compact index** – raw `uint8` histograms, 321 bytes/molecule (the fuzzy smoothing is never stored).
* **Exact search** – cosine / continuous-Tanimoto similarity in **[0, 1]** (distance = 1 − similarity), several queries per scan.

Tested on a 131M-molecule PubChem-derived set: full index built in 78 min, one exact query scans all 131M molecules in ~17 s on a laptop-class CPU (see [Performance](#performance)).

---

## The fingerprint (ErG v2)

Each molecule is reduced to *sites*; every pair of sites is binned by the shortest bond-count path between them.

| Site | Definition |
|---|---|
| `AR` | fully aromatic 5-, 6- or 7-membered ring (one site per ring) |
| `HY` | connected group of ≥ 3 neutral, non-aromatic carbons whose neighbours are all carbons |
| `PLUS` / `MINUS` | atom with positive / negative formal charge |
| `HA` | neutral N, O, S, F, Cl, Br (charged atoms are `PLUS`/`MINUS` instead) |
| `HD` | N or O carrying ≥ 1 H (implicit or explicit). It is a *flip-flop* atom: it gives **two sites on the same atom, `HA` and `HD`**; the pair between the two is skipped |

* 6 **count** columns (`count_HA` … `count_MINUS`) + 21 unordered type pairs × 15 distance bins (`HA_HD_d1` … `MINUS_MINUS_d15`) = **321 columns**.
* Pair distance = minimum bond count over the member atoms (atoms of rings/HY groups count as one site, `a ≠ b`), clamped to [1, 15].
  Pairs across disconnected fragments (salts) are skipped.
* **Fuzzy spillover:** every pair adds 1 to its bin and 0.3 to each neighbouring bin (no spill below d1 / above d15).

### Differences from the original script (`erg_calc_fragments_topo.py`, "v1")

1. **HD was never assigned.** `HA` consumed every N/O/S first and the flip-flop SMARTS `[#7,#8][H]` needs explicit hydrogens, so all HD columns were zero. v2 uses the total H count, with flip-flop sites as above.
2. **Pairs were silently dropped.** The key was built with an alphabetical test (`f1 <= f2`), so `AR_HA`, `AR_HD` and `MINUS_PLUS` never matched a column. v2 uses a fixed 6×6 pair-type lookup.
3. RDKit `fr_*` fragment descriptors and the MW / LogP / HBD / HBA columns are not part of v2.

Column names and order are unchanged, but v1 values are **not** comparable with v2 for any column involving HA or HD
(columns for HY / AR / PLUS / MINUS-only pairs agree to float32 precision).

### Fuzzy smoothing is a linear post-step

Smoothing along the distance axis is a fixed banded 15×15 matrix **F** (1 on the diagonal, 0.3 on the neighbours), applied to the 21×15 block.
The generator therefore stores only the **raw integer histogram** (`uint8`), and `fuzzy_expand(raw)` reproduces the classic fuzzy vector exactly.
Because **F** is symmetric, ⟨F q, F d⟩ = ⟨F² q, d⟩: at search time only the *query* is smoothed (with F²), the database stays raw, and one
precomputed ‖F d‖² per molecule gives cosine and Tanimoto.

---

## Install

The pure-Python path needs only `rdkit`, `numpy`, `pandas`, `pyarrow`, `tqdm`.

The C++ kernel needs RDKit **headers**, which the pip/conda `rdkit` wheels do not ship. A conda-forge environment that works:

```bash
conda create -n erg_cpp -c conda-forge python=3.12 "rdkit=2024.09.4" "librdkit-dev=2024.09.4" \
      libboost-devel pybind11 numpy pandas pyarrow tqdm cmake ninja
```

**Windows** (MSVC Build Tools 2022): edit or override the env path, then build; the module is written to the repository root.

```bat
set ERG_CONDA_ENV=C:\path\to\envs\erg_cpp
cpp\build.bat
```

`CMakeLists.txt` has a non-MSVC branch (`-O3 -march=native`) for Linux/macOS; I have **not** tested it there.

> **Keep the RDKit version pinned.** The fingerprint depends on RDKit's sanitization and aromaticity perception; the C++ kernel and the Python reference
> were validated against each other on 2024.09.4. `erg_v2.py` falls back to Python automatically if `erg_kernel` cannot be imported.

---

## Usage

### Fingerprints

```bash
# CSV or parquet with a SMILES column  ->  fuzzy float32 columns (default) or search-ready raw uint8
python erg_v2.py molecules.csv -o out.parquet
python erg_v2.py molecules.csv --raw -o out_prefix          # out_prefix.raw.npy / .sqnorm.npy / .ids.parquet
python erg_v2.py molecules.csv --backend python -j 1        # force the Python reference
```

```python
import erg_v2 as E
raw  = E.raw_from_smiles("OCCN")          # uint8[321]
vec  = E.fuzzy_expand(raw)[0]             # float32[321], classic fuzzy values
raws, ok = E.raw_many(smiles_list, jobs=0)   # batch, C++ kernel if available (jobs=0: all cores)
```

### Build an index and search it

```bash
python erg_index.py build  --csv big.csv --smiles-col Charged_Smiles --id-col ID --out idx_dir [--shard-rows 4000000] [-j 16]
python erg_index.py info   --index idx_dir
python erg_index.py search --index idx_dir --smiles "COC(=O)c1ccccc1" --k 50 --metric cosine --out hits.csv
python erg_index.py search --index idx_dir --smiles-file queries.smi --metric tanimoto
```

* Builds are **resumable** (completed shards are skipped); molecules that fail to parse are not indexed.
* Metrics: `cosine` (default) and `tanimoto` (continuous: ⟨a,b⟩ / (‖a‖² + ‖b‖² − ⟨a,b⟩)), both in [0, 1] for these non-negative vectors; `dot` is raw.
* Pass several queries in one call – they share a single scan of the index.

Index layout: `meta.json`, and per shard `shard_NNNNN.raw.bin` (`uint8 [n, 321]`), `.sqnorm.npy` (`float32 [n]`), `.ids.parquet` (`id`, `smiles`, small row groups so hit look-up is cheap).

```python
from erg_index import Index
ix  = Index("idx_dir")
res = ix.search(["CCO", "c1ccccc1O"], k=20, metric="cosine")     # list of {global_row, similarity}
ids, smiles = ix.fetch(res[0]["global_row"])
```

---

## Performance

Measured on an Intel i7-1370P (6 P + 8 E cores, 20 threads), 64 GB RAM, NVMe SSD, Windows 11. Fingerprinting includes SMILES parsing.

| Fingerprint generation (60k ChEMBL molecules) | molecules / s |
|---|---:|
| Python reference, 1 core | 1,050 |
| Python reference, 8–16 processes | ~2,900 |
| **C++ kernel, 1 thread** | **5,700** |
| **C++ kernel, 20 threads** | **40,000** (54,000 on the PubChem set) |

| PubChem-derived set, 131,039,081 molecules | |
|---|---|
| Index build | 78 min (≈ 28–31k molecules/s), 0 parse failures |
| Index size | 41.5 GB (321 B/molecule + norms + SMILES parquet) |
| Exact search, 1 query, 16 workers | 17 s |
| Exact search, 16 queries in one scan | 27 s (~1.7 s/query) |

The single-query scan reads ~2.5 GB/s and looks I/O-bound; I did not separate a truly cold scan from a cached one. Thread scaling is sub-linear (~7× on 20 threads), probably the mixed P/E-core CPU.
Extrapolating the build rate to a 6-billion-molecule library is roughly 1.5–2.5 days on this machine (40–54k/s for the kernel alone, ~30k/s measured end-to-end for the index build); **this has not been run**.

---

## Validation

* `test_erg_v2.py` – hand-checked molecules (flip-flop HD, `AR_HA`, `PLUS_MINUS`), fuzzy expansion vs an explicit v1-style loop, query-only smoothing vs brute force on expanded vectors for cosine / Tanimoto / dot. Needs `ALL_chembl_ErG.parquet` (see [Data](#data)).
* `test_cpp.py` – C++ kernel vs the Python reference on edge cases (invalid, empty, salts, charged, single atom) and 20,000 ChEMBL molecules (0 mismatches), plus a thread-scaling benchmark.
* `test_index.py` – builds a 3-shard index from the first 300k rows of a CSV, checks exact top-k against brute force for all three metrics, resume behaviour, and ID/SMILES look-up.
* `profile_erg.py`, `bench_parallel.py`, `bench_search.py` – profiling and benchmarks used for the numbers above.

## Known limitations

* **Raw counts saturate at 255** (`uint8`). On ChEMBL this affected only the `d15` bin of a 160-heavy-atom peptide (p99.99 of all values is 20). Warnings are printed; switch to `uint16` if you need exactness for very large molecules.
* **Search is exact brute force** (linear in library size). An approximate index (IVF-PQ / HNSW) is the obvious next step for billion-scale libraries and is not implemented.
* **Duplicates in the source data are preserved.** The PubChem-derived CSV lists several charge states per ID with identical SMILES, so the same ID can appear several times with similarity 1.0 in hit lists. De-duplicate on `id` if you need one hit per compound.
* Molecules that differ only by a feature-less fragment (e.g. `CC.` + the same molecule) get identical fingerprints.
* The ChEMBL parquet generated with the original script contains **v1 values** and is out of date for HA/HD columns; regenerate it with `erg_v2.py` before using it as a reference.
* Windows-first: `cpp/build.bat` and the tested toolchain are Windows/MSVC.

## Data

Large files are not stored in this repository. They are archived on Zenodo: **`<ZENODO DOI / URL – to be added>`**.
Intended contents: the PubChem-derived input CSV, the prebuilt `pubchem_idx/` index (≈ 41.5 GB), and the ChEMBL parquet regenerated with v2.

## Repository layout

```
erg_v2.py        fingerprint (Python reference), fuzzy expansion, CLI
erg_index.py     sharded index: build / info / search   (add to the repo)
cpp/             C++ kernel (erg_kernel.cpp, CMakeLists.txt, build.bat)
test_erg_v2.py, test_cpp.py, test_index.py              (test_index.py: add to the repo)
profile_erg.py, bench_parallel.py, bench_search.py      (bench_search.py: add to the repo)
erg_calc_fragments_topo.py   original v1 script, kept for reference
```

## References

* Stiefl N., Watson I.A., Baumann K., Zaliani A. *ErG: 2D pharmacophore descriptions for scaffold hopping.* J. Chem. Inf. Model. **46**, 208–220 (2006).
* RDKit: <https://www.rdkit.org> – this is an independent topological implementation, **not** the RDKit `rdReducedGraphs` ErG, and values are not numerically identical to it.

## License

MIT – see `LICENSE`.
