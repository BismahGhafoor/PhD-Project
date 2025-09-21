# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import glob
import os
import warnings
import platform
import zipfile  # <-- NEW

warnings.simplefilter(action='ignore')

# =============================================================================
# User Input and Configuration
# =============================================================================
current_directory = '/rfs/LRWE_Proj88/bg205/DataAnalysis'
current_directory_hpc = '/rfs/LRWE_Proj88/bg205/DataAnalysis'

# Look for zipped clinical files; will fall back to .txt if needed
clinical_zip_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Clinical_*.zip"
clinical_txt_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Clinical_*.txt"

smoking_codes_excel = "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/GOLD_Codes_FZ.xlsx"
smoking_sheet_name = "Smok"

max_rows_limit = 20000  # unchanged
final_columns = ["patid", "eventdate", "medcode"]

# =============================================================================
# Helpers
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print(f"{'-'*60}")
    path = current_directory if platform.system() == 'Windows' else current_directory_hpc
    os.chdir(path)
    print(f"Changed directory to: {os.getcwd()}")
    print(f"{'-'*60}\n")

def read_files():
    """Return a sorted list of clinical files (.zip preferred, else .txt)."""
    files = sorted(glob.glob(clinical_zip_pattern))
    if not files:
        files = sorted(glob.glob(clinical_txt_pattern))
    assert files, f"No clinical files found for patterns:\n  {clinical_zip_pattern}\n  {clinical_txt_pattern}"
    exts = {os.path.splitext(f)[1].lower() for f in files}
    print(f"\nFound {len(files)} clinical files with extension(s): {exts}\n")
    return files

def read_tab_maybe_zip(path, usecols=None):
    """
    Read a tab-delimited file that might be a .zip containing a single .txt.
    Returns a DataFrame with dtype=str for all columns (we'll parse dates later).
    """
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not members:
                raise FileNotFoundError(f"No .txt found inside {path}")
            inner = members[0]  # first .txt (CPRD zips typically contain one)
            with zf.open(inner) as fh:
                df = pd.read_csv(fh, sep="\t", header=0, dtype=str, usecols=usecols)
            print(f"  -> read {inner}")
            return df
    else:
        return pd.read_csv(path, sep="\t", header=0, dtype=str, usecols=usecols)

# =============================================================================
# Smoking Extraction (Clinical Only)
# =============================================================================
def append_clinical_smoking(files, smoking_medcodes, final_columns, max_rows_limit=np.inf):
    tmp_records = []
    start = time.perf_counter()
    print("\nStarting processing of clinical files for smoking status...\n")

    # Only load columns we actually use (saves a lot of I/O)
    needed_cols = list(set(final_columns + ["patid", "eventdate", "medcode"]))

    for idx, filename in enumerate(files):
        print(f'Processing file {idx+1}/{len(files)}: {os.path.basename(filename)}')
        df = read_tab_maybe_zip(filename, usecols=needed_cols)
        print(f'  Total rows in file: {len(df)}')

        # Dates -> datetime; drop invalid
        df["eventdate"] = pd.to_datetime(df["eventdate"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["eventdate"])

        # Earliest eventdate per patient (within this file)
        index_df = df.groupby("patid", as_index=False)["eventdate"].min().rename(columns={"eventdate": "indexdate"})
        df = df.merge(index_df, on="patid", how="left")

        # Filter to smoking codes
        df_filtered = df[df["medcode"].isin(smoking_medcodes)]
        print(f'  Rows after filtering for smoking medcodes: {len(df_filtered)}')

        # Accumulate
        tmp_records.extend(df_filtered.to_dict('records'))

        elapsed = round((time.perf_counter() - start) / 60, 2)
        print(f'  Total accumulated rows so far: {len(tmp_records)} (Elapsed: {elapsed} mins)\n')

    final_df = pd.DataFrame.from_dict(tmp_records)

    # Keep only requested columns (if present)
    cols_out = [c for c in final_columns if c in final_df.columns]
    final_df = final_df[cols_out]

    memory_size = np.round(final_df.memory_usage(deep=True).sum() / (1024**2), 1)
    print(f"Final smoking status dataset: {len(final_df)} rows, Memory usage: {memory_size} MB")

    output_filename = "Clinical_SmokingStatus_all.txt.gz"
    final_df.to_csv(output_filename, sep="\t", index=False, compression="gzip", date_format='%d/%m/%Y')
    finish = time.perf_counter()
    print(f"File '{output_filename}' created in {round((finish - start)/60, 2)} mins.\n")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    change_directory(current_directory, current_directory_hpc)

    smoking_codes_df = pd.read_excel(smoking_codes_excel, sheet_name=smoking_sheet_name, dtype=str)
    smoking_medcodes = smoking_codes_df['medcode'].dropna().unique().tolist()
    print(f"Total smoking medcodes loaded: {len(smoking_medcodes)}")

    clinical_files = read_files()
    append_clinical_smoking(clinical_files, smoking_medcodes, final_columns, max_rows_limit)
