# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:40:20 2022
Updated May  2025 to handle .zip archives containing .txt extracts
"""
# =============================================================================
# Import Modules
# =============================================================================
import pandas as pd
import numpy as np
import time
import glob
import os
import warnings
import platform
import zipfile

warnings.simplefilter(action='ignore')

# =============================================================================
# User Input
# =============================================================================
current_directory       = '/rfs/LRWE_Proj88/bg205/DataAnalysis'
current_directory_hpc   = '/rfs/LRWE_Proj88/bg205/DataAnalysis'

# now point at the ZIPs, not the .txt
clinical_files_directory = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Clinical_*.zip"

therapy_files_directory  = "GOLD/Therapy/*.zip"
test_files_directory     = "GOLD/Test/*.zip"

clinical_code_directory  = "/rfs/LRWE_Proj88/bg205/DataAnalysis/Medcode_filtering/filtered_diabetes_codes.txt"
therapy_code_directory   = "final_codelist_gold_therapy.txt"
test_code_directory      = "final_codelist_gold_test.txt"

filter_clinical = True
filter_therapy  = False
filter_test     = False
max_rows_limit  = 20000  # np.inf for unlimited


# =============================================================================
# Helper to read a single .txt from inside a .zip
# =============================================================================
def read_zip_txt_file(zip_path):
    """Open the zip, find the first .txt, and read into a DataFrame."""
    with zipfile.ZipFile(zip_path, 'r') as z:
        txt_members = [name for name in z.namelist() if name.endswith('.txt')]
        if not txt_members:
            raise ValueError(f"No .txt files found in archive {zip_path}")
        if len(txt_members) > 1:
            print(f"Warning: multiple .txt in {os.path.basename(zip_path)}; reading '{txt_members[0]}'")
        with z.open(txt_members[0]) as f:
            return pd.read_csv(f, sep="\t", dtype=str)


# =============================================================================
# Directory & file utilities (unchanged)
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print(f"{'-'*60}")
    if platform.system() == 'Windows':
        path = current_directory
    elif platform.system() == 'Linux':
        path = current_directory_hpc
    else:
        raise OSError("Unsupported operating system")

    os.chdir(path)
    print(f"Changed directory to: {os.getcwd()}")
    print(f"{'-'*60}\n")


def read_codes(codelist_directory, filter_by_codes):
    if not filter_by_codes:
        return None
    codes = pd.read_csv(codelist_directory, sep='\t', dtype={'code': str})
    assert len(codes) > 0, 'No codes found in file'
    print('Total codes found:')
    for term, count in codes.terminology.value_counts().items():
        print(f"  {term}: {count}")
    if 'name' in codes.columns:
        print('\nCode types:')
        for name, cnt in codes.name.value_counts().items():
            print(f"  {name}: {cnt}")
    print(f"{'-'*40}\n")
    return codes


def read_files(files_directory):
    files = sorted(glob.glob(files_directory))
    assert files, f'No files found in {files_directory}'
    exts = {os.path.splitext(f)[1] for f in files}
    print(f"{'-'*40}")
    if len(exts) > 1:
        raise ValueError(f"Mixed file types in {files_directory}: {exts}")
    print(f"All files have extension: {exts.pop()}")
    print(f"{'-'*40}\n")
    return files


# =============================================================================
# Change to working directory
# =============================================================================
change_directory(current_directory, current_directory_hpc)


# =============================================================================
# Clinical
# =============================================================================
def append_clinical(files, filter_by_codes=False, codes=None, max_rows_limit=np.inf):
    tmp_records = []
    chunk_number = 1
    start = time.perf_counter()

    medcodes = codes[codes.terminology == 'medcode'] if filter_by_codes else None
    entities = codes[codes.terminology == 'enttype'] if filter_by_codes else None

    for idx, zippath in enumerate(files, start=1):
        print(f'----- File {idx}/{len(files)}: {os.path.basename(zippath)} -----')
        # read the inner txt directly
        df = read_zip_txt_file(zippath)
        print(f" Rows in this file: {len(df)}")

        if filter_by_codes:
            df = df[
                df.medcode.isin(medcodes.code) |
                df.enttype.isin(entities.code)
            ]

        tmp_records.extend(df.to_dict('records'))
        print(f" Accumulated rows: {len(tmp_records)}")
        print(f" Elapsed: {round((time.perf_counter()-start)/60,2)} min\n")

        if len(tmp_records) >= max_rows_limit or idx == len(files):
            out_df = pd.DataFrame(tmp_records)
            size_mb = out_df.memory_usage(deep=True).sum() / (1024**2)
            print('-'*40)
            print(f"Saving chunk {chunk_number}  ({size_mb:.1f} MB)...")
            out_df.to_csv(f"Cleaned_GOLD_Extract_Clinical_{chunk_number}.txt",
                          sep='\t', index=False, date_format='%d/%m/%Y')
            print(f"Chunk {chunk_number} done in {round((time.perf_counter()-start)/60,2)} min\n")
            tmp_records = []
            chunk_number += 1

    print(f"All Clinical done in {round((time.perf_counter()-start)/60,2)} min")
    print(f"{'-'*60}\n")


print(f"{'='*60}\nAppending Clinical Files\n{'='*60}\n")
clinical_codes = read_codes(clinical_code_directory, filter_clinical)
clinical_files = read_files(clinical_files_directory)
append_clinical(clinical_files, filter_clinical, clinical_codes, max_rows_limit)


# =============================================================================
# Therapy (no change needed if each .zip has exactly one .txt)
# =============================================================================
def append_therapy(files, filter_by_codes=False, prodcodes=None, max_rows_limit=np.inf):
    tmp_records = []
    chunk_number = 1
    start = time.perf_counter()

    for idx, filepath in enumerate(files, start=1):
        print(f'----- File {idx}/{len(files)}: {os.path.basename(filepath)} -----')
        df = pd.read_csv(filepath, sep="\t", dtype=str)  # pandas infers zip
        print(f" Rows in this file: {len(df)}")

        if filter_by_codes:
            df = df[df.prodcode.isin(prodcodes.code)]

        tmp_records.extend(df.to_dict('records'))
        print(f" Accumulated rows: {len(tmp_records)}")
        print(f" Elapsed: {round((time.perf_counter()-start)/60,2)} min\n")

        if len(tmp_records) >= max_rows_limit or idx == len(files):
            out_df = pd.DataFrame(tmp_records)
            size_mb = out_df.memory_usage(deep=True).sum() / (1024**2)
            print('-'*40)
            print(f"Saving chunk {chunk_number}  ({size_mb:.1f} MB)...")
            out_df.to_csv(f"Cleaned_GOLD_Extract_Therapy_{chunk_number}.txt",
                          sep='\t', index=False, date_format='%d/%m/%Y')
            print(f"Chunk {chunk_number} done in {round((time.perf_counter()-start)/60,2)} min\n")
            tmp_records = []
            chunk_number += 1

    print(f"All Therapy done in {round((time.perf_counter()-start)/60,2)} min")
    print(f"{'-'*60}\n")


print(f"{'='*60}\nAppending Therapy Files\n{'='*60}\n")
therapy_codes = read_codes(therapy_code_directory, filter_therapy)
therapy_files = read_files(therapy_files_directory)
append_therapy(therapy_files, filter_therapy, therapy_codes, max_rows_limit)


# =============================================================================
# Test (similarly unchanged)
# =============================================================================
def append_test(files, filter_by_codes=False, entities=None, max_rows_limit=np.inf):
    tmp_records = []
    chunk_number = 1
    start = time.perf_counter()

    for idx, filepath in enumerate(files, start=1):
        print(f'----- File {idx}/{len(files)}: {os.path.basename(filepath)} -----')
        df = pd.read_csv(filepath, sep="\t", dtype=str)
        print(f" Rows in this file: {len(df)}")

        if filter_by_codes:
            df = df[df.enttype.isin(entities.code)]

        tmp_records.extend(df.to_dict('records'))
        print(f" Accumulated rows: {len(tmp_records)}")
        print(f" Elapsed: {round((time.perf_counter()-start)/60,2)} min\n")

        if len(tmp_records) >= max_rows_limit or idx == len(files):
            out_df = pd.DataFrame(tmp_records)
            size_mb = out_df.memory_usage(deep=True).sum() / (1024**2)
            print('-'*40)
            print(f"Saving chunk {chunk_number}  ({size_mb:.1f} MB)...")
            out_df.to_csv(f"Cleaned_GOLD_Extract_Test_{chunk_number}.txt",
                          sep='\t', index=False, date_format='%d/%m/%Y')
            print(f"Chunk {chunk_number} done in {round((time.perf_counter()-start)/60,2)} min\n")
            tmp_records = []
            chunk_number += 1

    print(f"All Test done in {round((time.perf_counter()-start)/60,2)} min")
    print(f"{'-'*60}\n")


print(f"{'='*60}\nAppending Test Files\n{'='*60}\n")
test_codes = read_codes(test_code_directory, filter_test)
test_files = read_files(test_files_directory)
append_test(test_files, filter_test, test_codes, max_rows_limit)
