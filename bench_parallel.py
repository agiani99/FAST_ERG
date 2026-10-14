import time, pyarrow.parquet as pq, numpy as np, erg_v2 as e

if __name__ == "__main__":
    smi = next(pq.ParquetFile("ALL_chembl_ErG.parquet").iter_batches(batch_size=60000)).to_pandas().SMILES.tolist()
    t = time.perf_counter(); e.raw_many(smi[:200], jobs=8, chunk=25, progress=False); print(f"pool startup+200 mols, 8 jobs: {time.perf_counter()-t:.1f}s", flush=True)
    for j, ch in ((1, 5000), (8, 1000), (16, 1000)):
        t = time.perf_counter(); e.raw_many(smi, jobs=j, chunk=ch, progress=False); dt = time.perf_counter() - t
        print(f"jobs={j:2d} chunk={ch}: {dt:5.1f}s  {len(smi)/dt:7.0f} mol/s", flush=True)
    L = np.array([len(s) for s in smi]); print("mean SMILES len first 20k / all 60k:", L[:20000].mean().round(1), L.mean().round(1))
