# -*- coding: utf-8 -*-
"""
Extract smoking status from CPRD Aurum zipped observation files using medcodeids,
with Slurm array support (one ZIP per task), line-buffered logging, and merging.
"""

import os
import sys
import time
import glob
import gzip
import zipfile
import warnings
import platform
import pandas as pd
import numpy as np

warnings.simplefilter(action='ignore')

# ensure print() shows up promptly in .out (Python 3.9+)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

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

output_basename = "Aurum_Clinical_SmokingStatus"  # per-task files: *_task####.txt.gz
final_columns = ["patid", "obsdate", "medcodeid", "value"]

# =============================================================================
# Helpers
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print("-" * 60)
    # Prefer scratch on Linux compute nodes; otherwise use the regular path.
    path = current_directory_hpc if platform.system() == 'Linux' else current_directory
    if path and os.path.isdir(path):
        os.chdir(path)
        print(f"Changed directory to: {os.getcwd()}")
    else:
        # Don't crash if the path isn't mounted on this node.
        print(f"WARNING: directory not found or inaccessible: {path}. Staying in {os.getcwd()}")
    print("-" * 60 + "\n")

def parse_task_id():
    """Return int task id from CLI arg or SLURM_ARRAY_TASK_ID env var.
       Special: if first arg is 'merge', return the string 'merge'."""
    if len(sys.argv) > 1 and sys.argv[1].strip().lower() == "merge":
        return "merge"
    if len(sys.argv) > 1 and str(sys.argv[1]).strip() != "":
        try:
            return int(sys.argv[1])
        except ValueError:
            print(f"WARNING: invalid task id '{sys.argv[1]}'; ignoring.")
    env_tid = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_tid is not None:
        try:
            return int(env_tid)
        except ValueError:
            print(f"WARNING: invalid SLURM_ARRAY_TASK_ID '{env_tid}'; ignoring.")
    return None

def merge_per_task_outputs(basename, target_name="Aurum_Clinical_SmokingStatus_ALL.txt.gz",
                           expected=None, wait_secs=0, poll_every=15):
    """
    Merge gzipped per-task TSVs named '<basename>_task*.txt.gz' into a single gzipped TSV.
    Writes header once (from first file), skips headers for the rest.
    If 'expected' is set, optionally wait (poll) until that many files appear (max wait=wait_secs).
    """
    start = time.perf_counter()
    pattern = f"{basename}_task*.txt.gz"

    def list_files():
        return sorted(glob.glob(pattern))

    if expected is not None and expected > 0 and wait_secs > 0:
        print(f"[merge] Waiting for {expected} files matching {pattern} "
              f"(timeout {wait_secs}s, every {poll_every}s)...")
        deadline = time.time() + wait_secs
        while time.time() < deadline:
            files = list_files()
            if len(files) >= expected:
                break
            print(f"[merge] Found {len(files)}/{expected} so far...", flush=True)
            time.sleep(poll_every)

    files = list_files()
    assert files, f"[merge] No per-task files found matching {pattern}"

    if expected is not None:
        print(f"[merge] Found {len(files)} per-task files (expected {expected}). Proceeding.")

    print(f"[merge] Merging into {target_name} ...")
    out_rows = 0
    with gzip.open(target_name, "wt") as out:
        header_written = False
        for f in files:
            with gzip.open(f, "rt") as inp:
                for i, line in enumerate(inp):
                    if i == 0:
                        if not header_written:
                            out.write(line)
                            header_written = True
                        else:
                            # skip subsequent headers
                            continue
                    else:
                        out.write(line)
                        out_rows += 1

    elapsed = round((time.perf_counter() - start) / 60, 2)
    print(f"[merge] Wrote {target_name} with ~{out_rows} data lines in {elapsed} minutes.")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    change_directory(current_directory, current_directory_hpc)

    task_id = parse_task_id()

    # ---- Merge-only mode (run after array completes) -------------------------
    if task_id == "merge":
        exp = os.environ.get("EXPECTED_TASKS")
        wait = os.environ.get("MERGE_WAIT_SECS")
        expected = int(exp) if exp and exp.isdigit() else None
        wait_secs = int(wait) if wait and wait.isdigit() else 0
        merge_per_task_outputs(output_basename,
                               target_name="Aurum_Clinical_SmokingStatus_ALL.txt.gz",
                               expected=expected,
                               wait_secs=wait_secs,
                               poll_every=15)
        sys.exit(0)

    print(f"Using smoking CSV folder: {smoking_csv_folder}")
    print(f"Using Observation folder: {observation_zip_folder}")

    # ---- Load medcodeids from CSVs ----
    medcodeids = []
    for f in csv_files:
        full_path = os.path.join(smoking_csv_folder, f)
        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"Medcode CSV not found: {full_path}")
        df_codes = pd.read_csv(full_path, dtype=str, skiprows=2)
        df_codes.columns = [c.lower().strip() for c in df_codes.columns]
        col_candidates = [col for col in df_codes.columns if 'medcode' in col]
        if not col_candidates:
            raise ValueError(f"Could not find medcodeid column in {f}")
        medcodeids.extend(df_codes[col_candidates[0]].dropna().astype(str).str.strip().tolist())

    smoking_medcodeids = sorted(set(medcodeids))
    smoking_medcode_set = set(smoking_medcodeids)
    print(f"Loaded {len(smoking_medcodeids)} unique smoking medcodeids.")

    # ---- Discover all ZIPs ----
    all_zip_files = sorted(glob.glob(os.path.join(observation_zip_folder, "*.zip")))
    assert all_zip_files, f"No zip files found in {observation_zip_folder}"
    print(f"Discovered {len(all_zip_files)} observation zip(s).")

    # ---- Slurm array task slicing: exactly one ZIP per task ----
    if task_id is not None:
        if task_id < 0 or task_id >= len(all_zip_files):
            raise IndexError(f"Task id {task_id} out of range 0..{len(all_zip_files)-1}")
        zip_files = [all_zip_files[task_id]]
        output_filename = f"{output_basename}_task{task_id:04d}.txt.gz"
        print(f"Array task {task_id}: processing {os.path.basename(zip_files[0])}")
    else:
        zip_files = all_zip_files
        output_filename = f"{output_basename}_all.txt.gz"
        print(f"No task id provided; processing ALL {len(zip_files)} zip(s)")

    tmp_records = []
    start = time.perf_counter()

    print(f"\nProcessing {len(zip_files)} zipped file(s)...\n")

    for idx, zip_path in enumerate(zip_files, start=1):
        print(f"[{idx}/{len(zip_files)}] Reading ZIP: {os.path.basename(zip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            txt_members = [m for m in zip_ref.namelist() if m.lower().endswith(".txt")]
            if not txt_members:
                print("  (No .txt members found in this zip)")
            for file_name in txt_members:
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

                keep_cols = [c for c in final_columns if c in df.columns]
                if len(keep_cols) < len(final_columns):
                    missing = [c for c in final_columns if c not in df.columns]
                    print(f"    WARNING: missing columns {missing} in '{file_name}'. Keeping {keep_cols}.")

                df_match = df.loc[matches, keep_cols]
                if not df_match.empty:
                    tmp_records.append(df_match)

    if not tmp_records:
        raise ValueError("Still no smoking-related rows found across the processed zips. "
                         "Check medcodeids and column names.")

    # ---- Combine, clean, annotate ----
    final_df = pd.concat(tmp_records, ignore_index=True)

    if 'obsdate' not in final_df.columns:
        raise KeyError("Column 'obsdate' not found in filtered data; cannot continue.")
    final_df['obsdate'] = pd.to_datetime(final_df['obsdate'], errors='coerce', dayfirst=True)
    final_df = final_df.dropna(subset=['obsdate'])

    if 'patid' not in final_df.columns:
        raise KeyError("Column 'patid' not found in filtered data; cannot continue.")
    index_df = final_df.groupby("patid", as_index=False)["obsdate"].min().rename(columns={"obsdate": "indexdate"})
    final_df = final_df.merge(index_df, on="patid", how="left")

    if 'medcodeid' in final_df.columns:
        final_df = final_df.rename(columns={"obsdate": "eventdate", "medcodeid": "medcode"})
    else:
        final_df = final_df.rename(columns={"obsdate": "eventdate"})

    mem_usage = np.round(final_df.memory_usage(deep=True).sum() / (1024**2), 1)
    print(f"Final dataset: {len(final_df)} rows, ~{mem_usage} MB")

    # ---- Save (TSV!) ----
    final_df.to_csv(output_filename, sep="\t", index=False, compression="gzip", date_format='%d/%m/%Y')
    elapsed = round((time.perf_counter() - start) / 60, 2)
    print(f"\nSaved '{output_filename}' in {elapsed} minutes.")

    # ---- Optional auto-merge from task 0 ------------------------------------
    # If MERGE_AFTER_ARRAY=1 and this is task 0, try to merge per-task outputs.
    if os.environ.get("MERGE_AFTER_ARRAY") == "1" and task_id == 0:
        exp = os.environ.get("EXPECTED_TASKS")
        wait = os.environ.get("MERGE_WAIT_SECS")
        expected = int(exp) if exp and exp.isdigit() else len(all_zip_files)
        wait_secs = int(wait) if wait and wait.isdigit() else 0  # default: no wait
        print(f"[merge-trigger] task 0 finishing; attempting merge (expected={expected}, wait_secs={wait_secs})")
        merge_per_task_outputs(output_basename,
                               target_name="Aurum_Clinical_SmokingStatus_ALL.txt.gz",
                               expected=expected,
                               wait_secs=wait_secs,
                               poll_every=15)
