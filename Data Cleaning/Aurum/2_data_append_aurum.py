# -*- coding: utf-8 -*-
"""
Aurum Data Cleaning Script (Updated to handle .zip archives)
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
current_directory = '/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM'
current_directory_hpc = '/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM'

clinical_files_directory = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/Aurum/Observation/FZ_Aurum_1_Extract_Observation_*.zip"
therapy_files_directory = "Aurum/DrugIssue/*.zip"

clinical_code_directory = "filtered_diabetes_AURUM_codes.txt"
therapy_code_directory = "final_codelist_gold_therapy.txt"

filter_clinical = True
filter_therapy = False
max_rows_limit = 20000

# =============================================================================
# Functions
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print("-" * 60)
    if platform.system() == 'Windows':
        path = current_directory
    elif platform.system() == 'Linux':
        path = current_directory_hpc
    else:
        raise OSError("Unsupported operating system")

    try:
        os.chdir(path)
        print(f"Changed directory to: {os.getcwd()}")
    except FileNotFoundError:
        print(f"Error: The path '{path}' does not exist.")
    print("-" * 60)

def read_codes(codelist_directory, filter_by_codes):
    if filter_by_codes:
        codes = pd.read_csv(codelist_directory, sep='\t', dtype=str)
        if 'medcodeid' in codes.columns:
            codes.rename(columns={'medcodeid': 'code'}, inplace=True)
        if 'productcodeid' in codes.columns:
            codes.rename(columns={'productcodeid': 'code'}, inplace=True)
        if 'termtype' in codes.columns:
            codes.rename(columns={'termtype': 'terminology'}, inplace=True)
        assert len(codes) > 0, 'No codes found in file'
        return codes

def read_files(files_directory):
    files = sorted(glob.glob(files_directory))
    assert len(files) > 0, f'No files found in {files_directory}'
    return files

def read_zip_file(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        name = z.namelist()[0]  # Assumes one file per zip
        with z.open(name) as f:
            df = pd.read_csv(f, sep='\t', dtype=str)
    return df

# =============================================================================
# Clinical (Observation)
# =============================================================================
def append_clinical(files, filter_by_codes=False, codes=None, max_rows_limit=np.inf):
    tmp_df = []
    chunk_number = 1
    start = time.perf_counter()

    medcodes = codes[codes['terminology'] == 'medcode']
    entities = codes[codes['terminology'] == 'enttype']

    for idx, filename in enumerate(files):
        df = read_zip_file(filename)
        print(f'Processed file {idx+1}: {os.path.basename(filename)} ({len(df)} rows)')

        if filter_by_codes:
            if "enttype" in df.columns:
                df = df[df["medcodeid"].isin(medcodes['code']) | df["enttype"].isin(entities['code'])]
            else:
                df = df[df["medcodeid"].isin(medcodes['code'])]

        tmp_df.extend(df.to_dict('records'))

        if len(tmp_df) >= max_rows_limit or (idx + 1 == len(files)):
            df_out = pd.DataFrame.from_dict(tmp_df)
            df_out.to_csv(f"Cleaned_AURUM_Observation_{chunk_number}.txt", sep='\t', index=False)
            tmp_df = []
            chunk_number += 1

    print(f"All clinical files completed in {round((time.perf_counter() - start)/60, 2)} mins")

# =============================================================================
# Therapy (DrugIssue)
# =============================================================================
def append_therapy(files, filter_by_codes=False, codes=None, max_rows_limit=np.inf):
    tmp_df = []
    chunk_number = 1
    start = time.perf_counter()

    for idx, filename in enumerate(files):
        df = read_zip_file(filename)
        print(f'Processed file {idx+1}: {os.path.basename(filename)} ({len(df)} rows)')

        if filter_by_codes:
            df = df[df["productcodeid"].isin(codes['code'])]

        tmp_df.extend(df.to_dict('records'))

        if len(tmp_df) >= max_rows_limit or (idx + 1 == len(files)):
            df_out = pd.DataFrame.from_dict(tmp_df)
            df_out.to_csv(f"Cleaned_AURUM_DrugIssue_{chunk_number}.txt", sep='\t', index=False)
            tmp_df = []
            chunk_number += 1

    print(f"All therapy files completed in {round((time.perf_counter() - start)/60, 2)} mins")

# =============================================================================
# Run
# =============================================================================
change_directory(current_directory, current_directory_hpc)

print("="*60 + "\nAppending Clinical Files\n" + "="*60)
clinical_codes = read_codes(clinical_code_directory, filter_clinical)
clinical_files = read_files(clinical_files_directory)
append_clinical(clinical_files, filter_clinical, clinical_codes, max_rows_limit)

print("="*60 + "\nAppending Therapy Files\n" + "="*60)
therapy_codes = read_codes(therapy_code_directory, filter_therapy)
therapy_files = read_files(therapy_files_directory)
append_therapy(therapy_files, filter_therapy, therapy_codes, max_rows_limit)
