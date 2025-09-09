# -*- coding: utf-8 -*-
"""
Extract smoking status from CPRD Aurum zipped observation files using medcodeids (with debug output).
"""

import pandas as pd
import numpy as np
import time
import os
import zipfile
import glob
import warnings
import platform

warnings.simplefilter(action='ignore')

# =============================================================================
# Configuration
# =============================================================================
current_directory = '/rfs/LRWE_Proj88/bg205/DataAnalysis'
current_directory_hpc = '/rfs/LRWE_Proj88/bg205/DataAnalysis'

observation_zip_folder = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/Aurum/Observation"
smoking_csv_folder = "/rfs/LRWE_Proj88/bg205/Codes/smoking_CSV_exports"
csv_files = [
    "Current_smoker.csv",
    "Ex-smoker.csv",
    "Never_smoked.csv"
]

output_filename = "Aurum_Clinical_SmokingStatus_all.txt.gz"
final_columns = ["patid", "obsdate", "medcodeid", "value"]


# =============================================================================
# Helper
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print(f"{'-'*60}")
    if platform.system() == 'Windows':
        path = current_directory
    elif platform.system() == 'Linux':
        path = current_directory_hpc
    else:
        raise OSError("Unsupported OS")
    os.chdir(path)
    print(f"Changed directory to: {os.getcwd()}")
    print(f"{'-'*60}\n")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    change_directory(current_directory, current_directory_hpc)

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

    smoking_medcodeids = list(set(medcodeids))
    print(f"Loaded {len(smoking_medcodeids)} unique smoking medcodeids.")

    zip_files = sorted(glob.glob(os.path.join(observation_zip_folder, "*.zip")))
    assert zip_files, f"No zip files found in {observation_zip_folder}"

    tmp_records = []
    start = time.perf_counter()

    print(f"\nProcessing {len(zip_files)} zipped files...\n")

    for idx, zip_path in enumerate(zip_files):
        print(f"({idx + 1}/{len(zip_files)}) Reading: {os.path.basename(zip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.lower().endswith(".txt"):
                    with zip_ref.open(file_name) as obs_file:
                        df = pd.read_csv(obs_file, sep="\t", dtype=str)
                        if 'medcodeid' not in df.columns:
                            print(f"No 'medcodeid' column in file {file_name}")
                            continue

                        df['medcodeid'] = df['medcodeid'].astype(str).str.strip()
                        match_count = df['medcodeid'].isin(smoking_medcodeids).sum()

                        print(f"  ➤ Matches in this file: {match_count}")
                        print("    Example medcodeids from data: ", df['medcodeid'].dropna().unique()[:5].tolist())
                        print("    First 5 smoking medcodeids: ", smoking_medcodeids[:5])

                        df = df[df['medcodeid'].isin(smoking_medcodeids)]
                        if not df.empty:
                            tmp_records.append(df[final_columns])

    if not tmp_records:
        raise ValueError("Still no smoking-related rows found across any zipped file. Check medcodeids and column names.")

    # Combine, clean, and save
    final_df = pd.concat(tmp_records, ignore_index=True)
    final_df['obsdate'] = pd.to_datetime(final_df['obsdate'], errors='coerce', dayfirst=True)
    final_df = final_df.dropna(subset=['obsdate'])

    index_df = final_df.groupby("patid", as_index=False)["obsdate"].min().rename(columns={"obsdate": "indexdate"})
    final_df = final_df.merge(index_df, on="patid", how="left")

    final_df = final_df.rename(columns={"obsdate": "eventdate", "medcodeid": "medcode"})

    mem_usage = np.round(final_df.memory_usage(deep=True).sum() / (1024**2), 1)
    print(f"Final dataset: {len(final_df)} rows, ~{mem_usage} MB")

    final_df.to_csv(output_filename, sep="\t", index=False, compression="gzip", date_format='%d/%m/%Y')
    elapsed = round((time.perf_counter() - start) / 60, 2)
    print(f"\nSaved '{output_filename}' in {elapsed} minutes.")
