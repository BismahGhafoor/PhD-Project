# -*- coding: utf-8 -*-
"""
Aurum Data Cleaning Script (reads .zip archives; writes chunks to filtered_aurum_chunks)
- Filters Observation (clinical) rows by medcode (and enttype if present)
- Optionally filters DrugIssue (therapy) rows by productcodeid
- Writes outputs to: /rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM/filtered_aurum_chunks
"""

import pandas as pd
import numpy as np
import time
import glob
import os
import zipfile
import warnings
import platform

warnings.simplefilter(action='ignore')

# =============================================================================
# User Input
# =============================================================================
# Working directory (where logs etc. will print from; not where chunks are written)
current_directory = '/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM'
current_directory_hpc = '/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM'

# INPUT globs
clinical_files_directory = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/Aurum/Observation/FZ_Aurum_1_Extract_Observation_*.zip"
therapy_files_directory  = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/Aurum/DrugIssue/*.zip"

# CODELISTS
clinical_code_directory = "filtered_diabetes_AURUM_codes.txt"   # produced by Script 1
therapy_code_directory  = "final_codelist_gold_therapy.txt"     # if used

# FILTER SWITCHES
filter_clinical = True
filter_therapy  = False   # set True only if you have productcode codelist

OUTPUT_DIR = "/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM/filtered_aurum_chunks"

# =============================================================================
# Functions
# =============================================================================
def change_directory(local_dir, hpc_dir=None):
    print("-" * 60)
    if platform.system() == 'Windows':
        path = local_dir
    elif platform.system() == 'Linux':
        path = hpc_dir or local_dir
    else:
        raise OSError("Unsupported operating system")
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # ensure output folder exists
    os.chdir(path)
    print(f"Changed directory to: {os.getcwd()}")
    print(f"Output directory:     {OUTPUT_DIR}")
    print("-" * 60)

def read_codes(codelist_path, do_filter):
    """
    Reads a codelist and normalizes to columns: ['terminology','code'] if possible.
    For clinical: expect 'medcodeid' → 'code', and optionally 'termtype' → 'terminology'.
    For therapy:  expect 'productcodeid' → 'code'.
    """
    if not do_filter:
        return None
    codes = pd.read_csv(codelist_path, sep='\t', dtype=str)
    # Normalize column names if present
    colmap = {}
    if 'medcodeid' in codes.columns:
        colmap['medcodeid'] = 'code'
    if 'productcodeid' in codes.columns:
        colmap['productcodeid'] = 'code'
    if 'termtype' in codes.columns:
        colmap['termtype'] = 'terminology'
    if colmap:
        codes = codes.rename(columns=colmap)
    # If terminology missing but we know it's medcodes (e.g. Script 1), fill it
    if 'terminology' not in codes.columns and 'code' in codes.columns:
        codes['terminology'] = 'medcode'
    assert len(codes) > 0, f'No codes found in {codelist_path}'
    return codes[['terminology','code']].dropna()

def read_zip_all_txt(zip_path):
    """
    Read and concatenate *all* TXT members in a ZIP into one DataFrame (dtype=str).
    Returns empty DF if none could be read.
    """
    frames = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if not name.lower().endswith(".txt"):
                continue
            with z.open(name) as f:
                try:
                    df = pd.read_csv(f, sep='\t', dtype=str, low_memory=False)
                    frames.append(df)
                except Exception as e:
                    print(f"  Skipping {name} in {os.path.basename(zip_path)} (read error: {e})")
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

def read_files(files_glob):
    files = sorted(glob.glob(files_glob))
    assert len(files) > 0, f'No files found for pattern: {files_glob}'
    return files

# =============================================================================
# Clinical (Observation)
# =============================================================================
def append_clinical(files, do_filter=False, codes=None):
    """
    Writes one output per input ZIP:
      OUTPUT_DIR/Cleaned_AURUM_Observation_<seq>.txt
    Filtering: medcode (and enttype if present & supplied in codelist).
    """
    start = time.perf_counter()
    # Split codes into medcodes and enttypes if provided
    medcodes = pd.Series(dtype=str)
    entities = pd.Series(dtype=str)
    if do_filter and codes is not None:
        medcodes = codes.loc[codes['terminology'].str.lower()=='medcode', 'code'].dropna().astype(str)
        if 'enttype' in codes['terminology'].str.lower().unique():
            entities = codes.loc[codes['terminology'].str.lower()=='enttype', 'code'].dropna().astype(str)

    out_count = 0
    for idx, zipf in enumerate(files, start=1):
        df = read_zip_all_txt(zipf)
        print(f"[Clinical] {idx}/{len(files)}: {os.path.basename(zipf)} -> {len(df):,} rows before filter")

        if do_filter and not df.empty:
            if "medcodeid" in df.columns:
                if "enttype" in df.columns and not entities.empty:
                    df = df[
                        df["medcodeid"].astype(str).isin(medcodes) |
                        df["enttype"].astype(str).isin(entities)
                    ]
                else:
                    df = df[df["medcodeid"].astype(str).isin(medcodes)]
            else:
                print("  Warning: 'medcodeid' not in columns; no clinical filtering applied.")

        out_count += 1
        out_path = os.path.join(OUTPUT_DIR, f"Cleaned_AURUM_Observation_{out_count}.txt")
        df.to_csv(out_path, sep='\t', index=False)
        print(f"  -> Wrote {len(df):,} rows to {out_path}")

    print(f"All clinical files completed in {round((time.perf_counter() - start)/60, 2)} mins")

# =============================================================================
# Therapy (DrugIssue)
# =============================================================================
def append_therapy(files, do_filter=False, codes=None):
    """
    Writes one output per input ZIP:
      OUTPUT_DIR/Cleaned_AURUM_DrugIssue_<seq>.txt
    If do_filter=True, keep rows with productcodeid in codes['code'].
    """
    start = time.perf_counter()
    prodcodes = pd.Series(dtype=str)
    if do_filter and codes is not None:
        prodcodes = codes['code'].dropna().astype(str)

    out_count = 0
    for idx, zipf in enumerate(files, start=1):
        df = read_zip_all_txt(zipf)
        print(f"[Therapy]  {idx}/{len(files)}: {os.path.basename(zipf)} -> {len(df):,} rows before filter")

        if do_filter and not df.empty:
            if "productcodeid" in df.columns:
                df = df[df["productcodeid"].astype(str).isin(prodcodes)]
            else:
                print("  Warning: 'productcodeid' not in columns; no therapy filtering applied.")

        out_count += 1
        out_path = os.path.join(OUTPUT_DIR, f"Cleaned_AURUM_DrugIssue_{out_count}.txt")
        df.to_csv(out_path, sep='\t', index=False)
        print(f"  -> Wrote {len(df):,} rows to {out_path}")

    print(f"All therapy files completed in {round((time.perf_counter() - start)/60, 2)} mins")

# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    change_directory(current_directory, current_directory_hpc)

    print("="*60 + "\nAppending Clinical Files\n" + "="*60)
    clinical_codes = read_codes(clinical_code_directory, filter_clinical)
    clinical_files = read_files(clinical_files_directory)
    append_clinical(clinical_files, filter_clinical, clinical_codes)

    print("="*60 + "\nAppending Therapy Files\n" + "="*60)
    therapy_codes = read_codes(therapy_code_directory, filter_therapy)
    therapy_files = read_files(therapy_files_directory)
    append_therapy(therapy_files, filter_therapy, therapy_codes)
