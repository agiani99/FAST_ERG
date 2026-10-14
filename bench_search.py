import time, sys, numpy as np, pandas as pd
import erg_index as X

if __name__ == "__main__":
    ix = X.Index("pubchem_idx")
    df = pd.read_csv("comprehensive_mw2000_collection.csv", usecols=["Charged_Smiles", "ID"], skiprows=range(1, 60_000_001), nrows=20)
    qs = df.Charged_Smiles.tolist()
    print(f"index: {ix.n:,} molecules", flush=True)
    for label, nq, workers in (("1 query,  8 workers", 1, 8), ("1 query, 16 workers", 1, 16), ("16 queries, 16 workers", 16, 16)):
        t = time.time(); res = ix.search(qs[:nq], k=20, metric="cosine", workers=workers); dt = time.time() - t
        print(f"{label}: {dt:6.1f}s  = {ix.n*nq/dt/1e6:7.1f} M mol-queries/s   (self-hit sim {res[0]['similarity'][0]:.4f})", flush=True)
    # show an actual result
    r = ix.search(qs[:1], k=8, metric="tanimoto", workers=16)[0]
    ids, smi = ix.fetch(r["global_row"])
    print("\nquery:", qs[0])
    for i, s, sim in zip(ids, smi, r["similarity"]):
        print(f"  {sim:.4f}  {i}  {s}")
