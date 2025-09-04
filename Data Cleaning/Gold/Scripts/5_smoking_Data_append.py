# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 11:40:20 2022
Modified for Smoking Status extraction WITHOUT additional clinical merge
Only keeps patid, eventdate, medcode, and indexdate in the final output
@author: ss1279
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

warnings.simplefilter(action='ignore')

# =============================================================================
# User Input and Configuration
# =============================================================================
# Directories for running on different systems
current_directory = '/rfs/LRWE_Proj88/bg205/DataAnalysis'
current_directory_hpc = '/rfs/LRWE_Proj88/bg205/DataAnalysis'

# Pattern for clinical files (raw extract with tab-delimited values)
clinical_files_directory = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Clinical_*.txt"

# Excel file containing smoking medcodes.
# The Excel file should have a sheet named "Smok" that includes the smoking codes.
# It is assumed that the medcodes are stored in a column named 'medcode'.
smoking_codes_excel = "GOLD_Codes_FZ.xlsx"
smoking_sheet_name = "Smok"

# Maximum number of rows to accumulate before writing out
max_rows_limit = 20000  # adjust as needed

# Final columns to keep in output
final_columns = ["patid", "eventdate", "medcode"]

# =============================================================================
# Helper Functions
# =============================================================================
def change_directory(current_directory, current_directory_hpc=None):
    print(f"{'-'*60}")
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
    except Exception as e:
        print(f"An error occurred: {e}")
    print(f"{'-'*60}\n")

def read_files(files_directory):
    files = sorted(glob.glob(files_directory))
    assert (len(files) > 0), f'No files found in {files_directory}'
    file_extensions = {os.path.splitext(file)[1] for file in files}
    print(f"{'-'*40}\n")
    if len(file_extensions) > 1:
        raise ValueError(f"Mixed file types in {files_directory}:{file_extensions}")
    else:
        ext = file_extensions.pop()
        print(f"All files have the same extension: {ext}")
    return files

# =============================================================================
# Smoking Extraction (Clinical Only)
# =============================================================================
def append_clinical_smoking(files, smoking_medcodes, final_columns, max_rows_limit=np.inf):
    """
    Processes clinical files by:
      1. Converting eventdate to datetime and dropping invalid dates.
      2. Computing the indexdate (earliest eventdate per patient) and merging it.
      3. Filtering for rows with a medcode in the smoking_medcodes list.
      4. Selecting only final_columns in the output
      5. Saving all matching rows to a gzipped text file.
    """
    tmp_records = []
    start = time.perf_counter()
    print("\nStarting processing of clinical files for smoking status...\n")
    
    for idx, filename in enumerate(files):
        # --- Read the clinical file ---
        df = pd.read_csv(filename, sep="\t", header=0, dtype=str)
        print(f'Processing file {idx+1}: {os.path.basename(filename)}')
        print(f'  Total rows in file: {len(df)}')
        
        # --- Convert eventdate column to datetime (dayfirst=True) and drop invalid dates ---
        df["eventdate"] = pd.to_datetime(df["eventdate"], errors="coerce", dayfirst=True)
        df = df.dropna(subset=["eventdate"])
        
        # --- Compute the indexdate for each patient (minimum eventdate) ---
        index_df = df.groupby("patid", as_index=False)["eventdate"].min().rename(columns={"eventdate": "indexdate"})
        df = df.merge(index_df, on="patid", how="left")
        
        # --- Filter rows for smoking medcodes ---
        df_filtered = df[df["medcode"].isin(smoking_medcodes)]
        print(f'  Rows after filtering for smoking medcodes: {len(df_filtered)}')
        
        # --- Accumulate records ---
        tmp_records.extend(df_filtered.to_dict('records'))
        
        elapsed = round((time.perf_counter() - start) / 60, 2)
        print(f'  Total accumulated rows so far: {len(tmp_records)} (Elapsed time: {elapsed} mins)\n')
    
    # --- After processing all files, create final DataFrame ---
    final_df = pd.DataFrame.from_dict(tmp_records)
    
    # --- Keep only the columns your supervisor wants ---
    # (Make sure they exist in final_df; otherwise you'll get a KeyError)
    columns_to_use = [c for c in final_columns if c in final_df.columns]
    final_df = final_df[columns_to_use]
    
    # --- Print memory usage info ---
    memory_size = np.round(final_df.memory_usage(deep=True).sum() / (1024**2), 1)
    print(f"Final smoking status dataset: {len(final_df)} rows, Memory usage: {memory_size} MB")
    
    # --- Write the final DataFrame to a gzipped text file ---
    output_filename = "Clinical_SmokingStatus_all.txt.gz"
    final_df.to_csv(output_filename, sep="\t", index=False, compression="gzip", date_format='%d/%m/%Y')
    finish = time.perf_counter()
    print(f"File '{output_filename}' created in {round((finish - start)/60, 2)} mins.\n")

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == '__main__':
    # Change to the appropriate working directory
    change_directory(current_directory, current_directory_hpc)
    
    # Load the smoking codes from the Excel file (sheet "Smok").
    try:
        smoking_codes_df = pd.read_excel(smoking_codes_excel, sheet_name=smoking_sheet_name, dtype=str)
    except Exception as e:
        raise FileNotFoundError(f"Error reading the Excel file {smoking_codes_excel}: {e}")
    
    # Drop missing values and obtain unique smoking medcodes as a list
    smoking_medcodes = smoking_codes_df['medcode'].dropna().unique().tolist()
    print(f"Total smoking medcodes loaded: {len(smoking_medcodes)}")
    
    # Read all clinical files using the defined pattern
    clinical_files = read_files(clinical_files_directory)
    
    # Process clinical files:
    # Filter by smoking medcodes, select only final columns, and write the final file.
    append_clinical_smoking(clinical_files, smoking_medcodes, final_columns, max_rows_limit)
