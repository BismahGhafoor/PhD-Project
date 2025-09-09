import pandas as pd
import glob
import os

chunks_dir = "/scratch/alice/b/bg205/smoking_run/smoking_tmp_outputs"
pattern = os.path.join(chunks_dir, "smoking_chunk_*.csv.gz")

files = sorted(glob.glob(pattern))
if not files:
    raise FileNotFoundError(f"No chunk files found matching {pattern}")

df_all = pd.concat((pd.read_csv(f, dtype=str) for f in files), ignore_index=True)
df_all.to_csv("Aurum_smoking_records_all.csv.gz", index=False, compression="gzip")
print(f"Wrote {len(df_all):,} rows to Aurum_smoking_records_all.csv.gz")
