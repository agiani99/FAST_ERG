import time, cProfile, pstats, io, numpy as np, pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdmolops
import erg_v2 as e
N = 5000
smi = next(pq.ParquetFile("ALL_chembl_ErG.parquet").iter_batches(batch_size=N)).to_pandas().SMILES.tolist()
def tm(f, *a):
    t = time.perf_counter(); r = f(*a); return time.perf_counter() - t, r
tp, mols = tm(lambda: [Chem.MolFromSmiles(s) for s in smi])
ts, sites = tm(lambda: [e.identify_sites(m) for m in mols])
td, _ = tm(lambda: [rdmolops.GetDistanceMatrix(m) for m in mols])
tt, _ = tm(lambda: [e.erg_raw(m) for m in mols])
ha = np.mean([m.GetNumHeavyAtoms() for m in mols]); ns = np.mean([len(s) for s in sites])
print(f"{N} mols, mean heavy atoms {ha:.1f}, mean sites {ns:.1f}, mean site pairs {np.mean([len(s)*(len(s)-1)/2 for s in sites]):.0f}")
for n, t in [("parse SMILES", tp), ("identify_sites", ts), ("GetDistanceMatrix", td), ("erg_raw total", tt)]:
    print(f"{n:20s} {t/N*1e6:8.0f} us/mol")
print(f"pair loop+rest ~ {(tt-ts-td)/N*1e6:.0f} us/mol;  total incl. parse {(tt+tp)/N*1e6:.0f} us/mol = {N/(tt+tp):.0f} mol/s/core")
pr = cProfile.Profile(); pr.enable(); [e.erg_raw(m) for m in mols[:1000]]; pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(8); print(s.getvalue()[:1800])
