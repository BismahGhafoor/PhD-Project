#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to create Test_entities_all.txt.gz from zipped or plain test extracts.

This script:
  1. Locates raw test files (prefers *.zip, falls back to *.txt).
  2. Reads each file in chunks (memory-friendly).
  3. Parses eventdate, drops invalid dates.
  4. Computes per-chunk indexdate = min(eventdate) per patid.
  5. Keeps patid, eventdate, indexdate, enttype, data1, data2.
  6. Renames data1->value, data2->unit.
  7. Appends to a gzipped output file.
"""

import pandas as pd
import glob
import os
import time

# ---------------------------
# User Input and Configuration
# ---------------------------
# Prefer zipped test files; fall back to txt if any remain unzipped
raw_zip_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Test_*.zip"
raw_txt_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Test_*.txt"

raw_test_files = sorted(glob.glob(raw_zip_pattern))
if not raw_test_files:
    raw_test_files = sorted(glob.glob(raw_txt_pattern))

print(f"Found {len(raw_test_files)} raw test files.")
if not raw_test_files:
    raise FileNotFoundError(f"No test files found for patterns:\n  {raw_zip_pattern}\n  {raw_txt_pattern}")

# Output filename for the combined, gzipped file
output_filename = "Test_entities_all.txt.gz"

# Remove the output file if it exists already
if os.path.exists(output_filename):
    os.remove(output_filename)
    print(f"Removed existing file: {output_filename}")

# Define chunk size (adjust as needed)
max_rows_limit = 20000

# Columns we expect to keep
cols_needed = ["patid", "eventdate", "indexdate", "enttype", "data1", "data2", "data3"]

# ---------------------------
# Processing Loop
# ---------------------------
start = time.perf_counter()

for idx, filename in enumerate(raw_test_files):
    print(f"\nProcessing file {idx+1} of {len(raw_test_files)}: {filename}")
    compression = "zip" if filename.lower().endswith(".zip") else "infer"

    reader = pd.read_csv(
        filename,
        sep="\t",
        dtype=str,
        chunksize=max_rows_limit,
        compression=compression,
        # If you know the input always includes these columns, you can uncomment:
        usecols=["patid", "eventdate", "enttype", "data1", "data2", "data3"],
        # (indexdate is computed below)
    )

    for chunk in reader:
        # Ensure required base columns exist
        base_required = ["patid", "eventdate"]
        for col in base_required:
            if col not in chunk.columns:
                raise KeyError(f"Required column '{col}' not found in {filename}")

        # Convert eventdate to datetime (dayfirst=True) and drop invalid
        chunk["eventdate"] = pd.to_datetime(chunk["eventdate"], errors="coerce", dayfirst=True)
        chunk = chunk.dropna(subset=["eventdate"])

        if chunk.empty:
            continue

        # Compute per-chunk indexdate = min(eventdate) per patid
        index_df = (
            chunk.groupby("patid", as_index=False)["eventdate"]
                 .min()
                 .rename(columns={"eventdate": "indexdate"})
        )
        chunk = chunk.merge(index_df, on="patid", how="left")

        # Select the necessary columns (take intersection to be safe)
        keep_cols = [c for c in cols_needed if c in chunk.columns]
        for missing in set(cols_needed) - set(keep_cols):
            chunk[missing] = pd.NA
            keep_cols.append(missing)
            
        chunk = chunk[keep_cols]

        chunk = chunk.rename(columns={
        "data2": "value",
        "data3": "unit"
        })
        # (optional) keep operator for QA, or drop:
        # chunk = chunk.drop(columns=["data1"])

        # Append the chunk to the output gz
        chunk.to_csv(
            output_filename,
            mode="a",
            header=not os.path.exists(output_filename),
            index=False,
            sep="\t",
            compression="gzip",
            date_format="%d/%m/%Y",
        )

    elapsed = round((time.perf_counter() - start) / 60, 2)
    print(f"Finished processing file: {filename}. Elapsed time: {elapsed} mins.")

print("\nTest_entities_all.txt.gz created successfully.")
