"""Validate + benchmark the C++ kernel against the python reference (erg_v2.raw_from_smiles).
Run inside the erg_cpp env:  python test_cpp.py"""
import time, numpy as np, pyarrow.parquet as pq
import erg_kernel, erg_v2 as e

if __name__ == "__main__":
    smi = next(pq.ParquetFile("ALL_chembl_ErG.parquet").iter_batches(batch_size=60000)).to_pandas().SMILES.tolist()

    # edge cases: invalid, empty, salt, charged, single atom, peptide-sized
    edge = ["", "not_a_smiles", "C", "[Na+].[Cl-]", "C[NH3+].CC(=O)[O-]", "OCCN", "c1ccccc1O", "CCCCCC",
            "[NH3+]CCCC(=O)[O-]", "O=C1NC(=O)c2ccccc12", "C1CCCCC1.C1CCCCC1"]
    r, ok = erg_kernel.raw_batch(edge, 1)
    for s, row, o in zip(edge, r, ok):
        ref = e.raw_from_smiles(s) if s else None
        same = (ref is None and not o) or (ref is not None and o and (ref == row).all())
        assert same, ("edge mismatch", s)
    print("edge cases identical to python reference:", len(edge))

    # full comparison on the first 20k
    n = 20000
    R, OK = erg_kernel.raw_batch(smi[:n], 1)
    P = np.array([e.raw_from_smiles(s) for s in smi[:n]])
    bad = np.where((R != P).any(1))[0]
    print(f"C++ == python on {n} ChEMBL molecules: {len(bad) == 0}  (mismatching rows: {len(bad)}; parse failures: {int((~OK).sum())})")
    for b in bad[:5]:
        print("  ", smi[b][:80], int((R[b] != P[b]).sum()), "cols differ")
    assert len(bad) == 0

    # speed (60k slice, parse included)
    for th in (1, 4, 8, 16, 20):
        t = time.perf_counter(); erg_kernel.raw_batch(smi, th); dt = time.perf_counter() - t
        print(f"C++ threads={th:2d}: {dt:6.2f}s  {len(smi) / dt:8.0f} mol/s", flush=True)
