# -*- coding: utf-8 -*-
"""
Extract smoking status from CPRD Aurum zipped observation files using medcodeids (with debug output).
One-zip-per-task: pass SLURM_ARRAY_TASK_ID or a numeric CLI arg. Task i reads only zip i.

Also supports: `python test_smoke.py merge`
→ merges Aurum_Clinical_SmokingStatus_task*.txt.gz into Aurum_Clinical_SmokingStatus_ALL.txt.gz
"""

import pandas as pd
import numpy as np
import time
import os
import zipfile
import glob
import warnings
import platform
import sys
import gzip

warnings.simplefilter(action='ignore')

# =============================================================================
# Configuration
# =============================================================================
current_directory = '/rfs/LRWE_Proj88/bg205/DataAnalysis'
current_directory_hpc = '/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum'

observation_zip_folder = "/scratch/alice/b/bg205/smoking_data_input/Observation"
smoking_csv_folder = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/Codes/smoking_CSV_exports"
csv_files = [
    "Current_smoker.csv",
    "Ex-smoker.csv",
    "Never_smoked.csv"
]

output_dir = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum"
output_basename = "Aurum_Clinical_SmokingStatus"   # chunks: *_task####.txt.gz
final_columns = ["patid", "obsdate", "medcodeid", "value"]

# =============================================================================
# Helpers
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print(f"{'-'*60}")
    if platform.system() == 'Windows':
        path = current_directory
    elif platform.system() == 'Linux':
        path = current_directory_hpc
    else:
        raise OSError("Unsupported OS")
    if path and os.path.isdir(path):
        os.chdir(path)
        print(f"Changed directory to: {os.getcwd()}")
    else:
        print(f"WARNING: directory not found or inaccessible: {path}. Staying in {os.getcwd()}")
    print(f"{'-'*60}\n")

def parse_task_id():
    """
    Get task id from CLI or SLURM_ARRAY_TASK_ID.
    Returns: int (0-based), "merge", or None.
    """
    # special CLI arg: 'merge'
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() == "merge":
        return "merge"
    # CLI arg numeric
    if len(sys.argv) > 1 and sys.argv[1].strip() != "":
        try:
            return int(sys.argv[1])
        except ValueError:
            print(f"WARNING: invalid task id '{sys.argv[1]}'; ignoring.")
    # Slurm env var
    env_tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_tid is not None:
        try:
            return int(env_tid)
        except ValueError:
            print(f"WARNING: invalid SLURM_ARRAY_TASK_ID '{env_tid}'; ignoring.")
    return None

def merge_per_task_outputs(out_dir, basename):
    """
    Merge TSV gzip chunks named '<basename>_task*.txt.gz' into
    '<basename>_ALL.txt.gz' in out_dir, streaming to keep memory low.
    """
    pattern = os.path.join(out_dir, f"{basename}_task*.txt.gz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No chunk files found matching {pattern}")

    target = os.path.join(out_dir, f"{basename}_ALL.txt.gz")
    print(f"[merge] Merging {len(files)} files -> {target}")

    wrote_header = False
    start = time.perf_counter()
    with gzip.open(target, "wt") as w:
        for f in files:
            for chunk in pd.read_csv(f, sep="\t", dtype=str, chunksize=500_000):
                chunk.to_csv(w, sep="\t", index=False, header=not wrote_header)
                wrote_header = True
    mins = round((time.perf_counter() - start) / 60, 2)
    print(f"[merge] Done in {mins} min")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    change_directory(current_directory, current_directory_hpc)

    task_id = parse_task_id()
    if task_id == "merge":
        merge_per_task_outputs(output_dir, output_basename)
        sys.exit(0)

    # Load medcodeids from CSVs
    medcodeids = []
    for f in csv_files:
        full_path = os.path.join(smoking_csv_folder, f)
        df = pd.read_csv(full_path, dtype=str, skiprows=2)
        df.columns = [c.lower().strip() for c in df.columns]
        col_candidates = [col for col in df.columns if 'medcode' in col]
        if not col_candidates:
            raise ValueError(f"Could not find medcodeid column in {f}")
        medcodeids.extend(df[col_candidates[0]].dropna().astype(str).str.strip().tolist())

    smoking_medcodeids = sorted(set(medcodeids))
    smoking_medcode_set = set(smoking_medcodeids)
    print(f"Loaded {len(smoking_medcodeids)} unique smoking medcodeids.")

    # Discover all ZIPs
    all_zip_files = sorted(glob.glob(os.path.join(observation_zip_folder, "*.zip")))
    assert all_zip_files, f"No zip files found in {observation_zip_folder}"
    print(f"Discovered {len(all_zip_files)} observation zip(s).")

    # Decide which ZIP(s) to process for this task
    if task_id is not None:
        if isinstance(task_id, int):
            if task_id < 0 or task_id >= len(all_zip_files):
                raise IndexError(f"Task id {task_id} out of range 0..{len(all_zip_files)-1}")
            zip_files = [all_zip_files[task_id]]
            output_filename = os.path.join(output_dir, f"{output_basename}_task{task_id:04d}.txt.gz")
            print(f"Array task {task_id}: processing {os.path.basename(zip_files[0])}")
        else:
            raise ValueError("Unexpected task_id state.")
    else:
        zip_files = all_zip_files
        output_filename = os.path.join(output_dir, f"{output_basename}_all.txt.gz")
        print(f"No task id provided; processing ALL {len(zip_files)} zip(s)")

    os.makedirs(output_dir, exist_ok=True)

    tmp_records = []
    start = time.perf_counter()

    print(f"\nProcessing {len(zip_files)} zipped file(s)...\n")

    for idx, zip_path in enumerate(zip_files, start=1):
        print(f"[{idx}/{len(zip_files)}] Reading ZIP: {os.path.basename(zip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in (m for m in zip_ref.namelist() if m.lower().endswith(".txt")):
                with zip_ref.open(file_name) as obs_file:
                    df = pd.read_csv(obs_file, sep="\t", dtype=str, low_memory=False)

                if 'medcodeid' not in df.columns:
                    print(f"  Skipping '{file_name}' (no 'medcodeid' column).")
                    continue

                df['medcodeid'] = df['medcodeid'].astype(str).str.strip()
                matches = df['medcodeid'].isin(smoking_medcode_set)
                match_count = int(matches.sum())

                print(f"  ➤ {file_name}: {match_count} matching rows")
                try:
                    print("    Example medcodeids from data:", df['medcodeid'].dropna().unique()[:5].tolist())
                    print("    First 5 smoking medcodeids:", smoking_medcodeids[:5])
                except Exception:
                    pass

                if match_count == 0:
                    continue

                # Ensure required columns present; create empties if missing
                for col in final_columns:
                    if col not in df.columns:
                        df[col] = pd.NA

                tmp_records.append(df.loc[matches, final_columns])

    if not tmp_records:
        raise ValueError("Still no smoking-related rows found across the processed zips. "
                         "Check medcodeids and column names.")

    # Combine, clean, and save
    final_df = pd.concat(tmp_records, ignore_index=True)
    final_df['obsdate'] = pd.to_datetime(final_df['obsdate'], errors='coerce', dayfirst=True)
    final_df = final_df.dropna(subset=['obsdate'])

    # Keep EXACT final columns + order
    final_df = final_df[final_columns]

    mem_usage = np.round(final_df.memory_usage(deep=True).sum() / (1024**2), 1)
    print(f"Final dataset: {len(final_df)} rows, ~{mem_usage} MB")

    # TSV gzip output
    final_df.to_csv(output_filename, sep="\t", index=False, compression="gzip", date_format='%d/%m/%Y')
    elapsed = round((time.perf_counter() - start) / 60, 2)
    print(f"\nSaved '{output_filename}' in {elapsed} minutes.")
