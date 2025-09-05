#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on [Your Date]
Script to create Test_entities_all.txt.gz

This script:
  1. Uses a glob pattern to locate raw test files (e.g. Test_Raw_*.txt).
  2. Reads each file in chunks (to conserve memory).
  3. Converts the eventdate column to datetime and drops invalid dates.
  4. Computes an indexdate for each patient (the minimum eventdate in the chunk).
  5. Selects a subset of columns (patid, eventdate, enttype, plus two lab‐related columns).
  6. Renames one lab column to “value” (the test result) and another to “unit” (the unit code).
  7. Appends each processed chunk to a gzipped output file.
  
Note:
  • This script assumes that the raw test file contains columns such as:
      patid, eventdate, sysdate, constype, consid, medcode, sctid, sctdescid,
      sctexpression, sctmaptype, sctmapversion, sctisindicative, sctisassured,
      staffid, enttype, data1, data2, data3, ... 
  • For lab tests the assumption here is that “data1” holds the test result (to be renamed as “value”)
    and “data2” holds the unit code (to be renamed as “unit”).
  • Adjust the raw_files_pattern and the column mapping if your raw files differ.
"""

import pandas as pd
import glob
import os
import time

# ---------------------------
# User Input and Configuration
# ---------------------------
# Change this pattern to point to your raw test files
raw_files_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Test_*.txt"  # <-- CHANGE THIS PATH
raw_test_files = sorted(glob.glob(raw_files_pattern))
print(f"Found {len(raw_test_files)} raw test files.")

# Output filename for the combined, gzipped file
output_filename = "Test_entities_all.txt.gz"

# Remove the output file if it exists already
if os.path.exists(output_filename):
    os.remove(output_filename)
    print(f"Removed existing file: {output_filename}")

# Define chunk size (adjust as needed)
max_rows_limit = 20000

# ---------------------------
# Processing Loop
# ---------------------------
start = time.perf_counter()

for idx, filename in enumerate(raw_test_files):
    print(f"\nProcessing file {idx+1} of {len(raw_test_files)}: {filename}")
    for chunk in pd.read_csv(filename, sep="\t", dtype=str, chunksize=max_rows_limit):
        # Convert eventdate to datetime (using dayfirst=True)
        chunk["eventdate"] = pd.to_datetime(chunk["eventdate"], errors="coerce", dayfirst=True)
        # Drop rows where eventdate is invalid
        chunk = chunk.dropna(subset=["eventdate"])
        
        # Compute indexdate for each patient as the minimum eventdate in this chunk
        index_df = chunk.groupby("patid", as_index=False)["eventdate"].min().rename(columns={"eventdate": "indexdate"})
        chunk = chunk.merge(index_df, on="patid", how="left")
        
        # Select the necessary columns.
        # Here we assume the raw file contains at least: patid, eventdate, enttype, data1, data2.
        # Adjust this list if your raw file is different.
        cols_needed = ["patid", "eventdate", "indexdate", "enttype", "data1", "data2"]
        chunk = chunk[cols_needed]
        
        # Rename data1 -> value and data2 -> unit so that lab extraction functions can use them
        chunk = chunk.rename(columns={"data1": "value", "data2": "unit"})
        
        # Append the chunk to the output file
        chunk.to_csv(output_filename, mode='a', 
                     header=not os.path.exists(output_filename),
                     index=False, sep="\t", compression="gzip", date_format='%d/%m/%Y')
    elapsed = round((time.perf_counter() - start) / 60, 2)
    print(f"Finished processing file: {filename}. Elapsed time: {elapsed} mins.")

print(f"\nTest_entities_all.txt.gz created successfully.")
