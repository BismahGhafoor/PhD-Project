#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on [Your Date]
Script for cleaning smoking, BMI and blood pressure data using all Additional Clinical files.

This script:
  1. Loads the Clinical Smoking Status file (gzipped).
  2. Processes ALL Additional Clinical Details Files in chunks WITHOUT loading all data into memory.
     Each chunk is filtered by enttype and appended to temporary CSV files.
  3. Loads patient data.
  4. Loads HES hospital data.
  5. Reads the subset files (bp, smoking, weight, height) into DataFrames.
  6. Ensures that each subset has the required date columns ("indexdate" and "eventdate"),
     merging them from the patient data if missing.
  7. Calls get_smoking_data (with smoking_data), weight_height_bmi (with concatenated weight and height data)
     and get_bp_data (with bp_data).
  8. Saves the final cleaned data to file.
  
Note: Adjust file paths and filenames as needed.
"""

import pandas as pd
import numpy as np
import glob
import os

print("testing hereee")

# ------------------------------------------------------------------------------
# Import helper functions used by get_smoking_data and long format saving
from helper_functions import lcf, ucf, perc
from helper_functions import save_long_format_data, read_long_format_data
from helper_functions import remap_eth, nperc_counts, calc_gfr

# Define save_long_format flag (adjust as needed)
save_long_format = False

# ------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ------------------------------------------------------------------------------
def get_smoking_data(smoking_data, clinical_smok, patient):
    # Ensure merging key 'patid' is string in all inputs.
    smoking_data['patid'] = smoking_data['patid'].astype(str)
    clinical_smok['patid'] = clinical_smok['patid'].astype(str)
    patient['patid'] = patient['patid'].astype(str)
    
    print("Columns in smoking_data before subsetting:", smoking_data.columns.tolist())
    try:
        smok_add = smoking_data[['patid', "indexdate", 'eventdate', 'data1']]
    except KeyError as e:
        print("ERROR: Missing expected columns in smoking_data:", e)
        raise

    # --- CHANGED: make data1 numeric before filtering 1/2/3
    smok_add["data1"] = pd.to_numeric(smok_add["data1"], errors="coerce")
    smok_add = smok_add[(smok_add['data1'].isin([1, 2, 3])) & (smok_add['eventdate'].notnull())]
    smok_add = smok_add.sort_values(['patid', 'eventdate']).reset_index(drop=True)
    smok_add = smok_add.sample(frac=1).drop_duplicates(subset=['patid', 'eventdate'], keep='last')
    smok_add['data1'] = smok_add['data1'].replace({1: "Yes", 2: "No", 3: "Ex"})
    nperc_counts(smok_add, 'data1')
    smok_add = smok_add.rename(columns={'data1': 'smok_add'})
    print("smok_add sample:")
    print(smok_add.head())
    smok_add.count()

    # --- Process clinical smoking (medcode) data ---
    smok_clref = clinical_smok.copy(deep=True)
    if "indexdate" not in smok_clref.columns:
        print("indexdate missing from clinical_smok; merging from patient.")
        smok_clref = smok_clref.merge(patient[['patid', 'indexdate']], on='patid', how='left')
    if "eventdate" not in smok_clref.columns:
        print("eventdate missing from clinical_smok; copying indexdate to eventdate.")
        smok_clref["eventdate"] = smok_clref["indexdate"]

    smok_clref['smok_clref'] = np.nan
    smoke_keys = ['current smoker', 'never smoker', 'ex smoker']
    smoke_cat = ['Yes', "No", 'Ex']
    filepath = '/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/GOLD_Codes_FZ.xlsx'
    smoke_codes = pd.read_excel(filepath, sheet_name="Smok")
    smoke_codes = smoke_codes[smoke_codes.source == 'cprd']
    smoke_values = []
    for key in smoke_keys:
        smoke_values.append(smoke_codes[smoke_codes['type'] == key]['medcode'].to_list())
    smoke_dict = dict(zip(smoke_keys, smoke_values))

    for idx, key in enumerate(smoke_keys):
        smok_clref.loc[smok_clref['medcode'].isin(smoke_dict.get(key, [])), 'smok_clref'] = smoke_cat[idx]
    smok_clref = smok_clref[smok_clref['smok_clref'].notnull()]
    smok_clref = smok_clref.sort_values(['patid', 'eventdate']).reset_index(drop=True)
    smok_clref = smok_clref.sample(frac=1).drop_duplicates(subset=['patid', 'eventdate'], keep='last')
    try:
        smok_clref = smok_clref[['patid', "indexdate", 'eventdate', 'smok_clref']]
    except KeyError as e:
        print("Error in smok_clref subsetting:", e)
        print("Columns in smok_clref:", smok_clref.columns.tolist())
        raise
    print("smok_clref sample:")
    print(smok_clref.head())
    smok_clref.count()

    # --- Process HES smoking data (ICD-10 based) ---
    filepath = '/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/GOLD_Codes_FZ.xlsx'
    smoke_codes = pd.read_excel(filepath, sheet_name="Smok")
    smoke_codes = smoke_codes[smoke_codes.source == 'hes']
    smoke_codes = smoke_codes.medcode.tolist()
    # Note: The variable hes_hosp must exist in the global scope.
    smok_hes = hes_hosp[hes_hosp["ICD"].fillna('').str.contains("|".join(smoke_codes))]
    smok_hes = smok_hes.sort_values(['patid', 'admidate']).reset_index(drop=True)
    smok_hes = smok_hes.sample(frac=1).drop_duplicates(subset=['patid', 'admidate'], keep='last')
    smok_hes['smok_hes'] = 'Yes'
    smok_hes = smok_hes.rename(columns={'admidate': 'eventdate'})
    # --- CHANGED: merge indexdate into smok_hes (not hes_hosp)
    if "indexdate" not in smok_hes.columns:
        print("indexdate missing from smok_hes; merging from patient.")
        smok_hes = smok_hes.merge(patient[['patid', 'indexdate']], on='patid', how='left')
    try:
        smok_hes = smok_hes[['patid', "indexdate", 'eventdate', 'smok_hes']]
    except KeyError as e:
        print("Error in smok_hes subsetting:", e)
        print("Columns in hes_hosp/smok_hes:", hes_hosp.columns.tolist())
        raise
    print("smok_hes sample:")
    print(smok_hes.head())
    smok_hes.count()

    # --- Merge all sources ---
    smoking = smok_clref.merge(smok_add, how='outer', on=['patid', "indexdate", 'eventdate'])
    smoking = smoking.merge(smok_hes, how='outer', on=['patid', "indexdate", 'eventdate'])
    smoking['smoking_status'] = smoking['smok_clref']
    smoking.loc[(smoking['smoking_status'].isnull()) & (smoking['smok_hes'].notnull()),
                'smoking_status'] = smoking['smok_hes']
    smoking.loc[(smoking['smoking_status'].isnull()) & (smoking['smok_add'].notnull()),
                'smoking_status'] = smoking['smok_add']
    nperc_counts(smoking, 'smoking_status')
    smoking = smoking.sort_values(['patid', 'eventdate']).reset_index(drop=True)
    smoking = smoking.sample(frac=1).drop_duplicates(subset=['patid', 'eventdate'], keep='last')
    smoking = smoking[['patid', 'indexdate', 'eventdate', 'smoking_status']]
    # (Date filtering removed; all records are kept.)
    save_long_format_data(smoking, save_long_format, 'smoking')
    smoking['smok_time_gap'] = (smoking['indexdate'] - smoking['eventdate']).abs().dt.days
    smoking = smoking.loc[smoking.groupby(['patid', 'indexdate'])['smok_time_gap'].idxmin()].reset_index(drop=True)
    smoking = smoking[['patid', 'indexdate', 'eventdate', 'smoking_status']]
    smoking = smoking.rename(columns={"eventdate": "smoking_date"})
    patient = patient.merge(smoking, on=['patid', 'indexdate'], how='left')
    return patient

def weight_height_bmi(wh_data, patient):
    # Use string comparisons for enttype values
    weight = wh_data[wh_data["enttype"] == "13"]
    weight = weight[["patid", "indexdate", "eventdate", "data1", "data3"]]
    weight = weight.rename(
        columns={
            "data1": "weight_kg",
            "data3": "bmi_recorded",
        }
    )

    height = wh_data[wh_data["enttype"] == "14"]
    height = height[["patid", "indexdate", "eventdate", "data1"]]
    height = height.rename(
        columns={
            "data1": "height_m",
        }
    )

    bmi = weight.merge(
        height, how="outer", on=["patid", "indexdate", "eventdate"]
    )
    float_cols = ["weight_kg", "height_m", "bmi_recorded"]
    bmi[float_cols] = bmi[float_cols].apply(pd.to_numeric, errors="coerce")

    bmi["height_m"] = bmi.groupby(["patid", "indexdate"])["height_m"].ffill()

    bmi["bmi_calc"] = bmi["weight_kg"] / (bmi["height_m"] * bmi["height_m"])

    bmi["bmi"] = bmi["bmi_recorded"]
    bmi.loc[(((bmi["bmi"].isin([0, np.inf])) | (bmi["bmi"].isna())) & (bmi["bmi_calc"] < np.inf)), "bmi"] = bmi["bmi_calc"]
    bmi = bmi[(bmi["bmi"] >= 10) & (bmi["bmi"] <= 70)]
    bmi.count()

    bmi = bmi.drop_duplicates(subset=['patid', 'indexdate', 'eventdate', 'bmi'], keep='last').reset_index(drop=True)
    bmi = bmi.groupby(['patid', 'indexdate', 'eventdate']).mean().reset_index()
    bmi = bmi.sort_values(["patid", "eventdate"]).reset_index(drop=True)
    bmi = bmi[["patid", "eventdate", "indexdate", "bmi"]]

    # Date filtering removed; all records are kept.
    save_long_format_data(bmi, save_long_format, 'bmi')

    bmi["bmi_time_gap"] = (bmi["indexdate"] - bmi["eventdate"]).abs().dt.days
    bmi = bmi.loc[bmi.groupby(["patid", "indexdate"])["bmi_time_gap"].idxmin()].reset_index(drop=True)

    bmi = bmi[["patid", "eventdate", "indexdate", "bmi"]]
    bmi = bmi.rename(columns={"eventdate": "bmi_date"})
    patient = patient.merge(bmi, on=["patid", "indexdate"], how="left")

    return patient

def get_bp_data(bp_data, patient):
    bp = bp_data[["patid", "eventdate", "indexdate", "data1", "data2"]]
    bp = bp.rename(columns={"data1": "diastolic", "data2": "systolic"})
    bp = bp[
        (
            (bp["eventdate"].notnull())
            & (bp["diastolic"].notnull())
            & (bp["systolic"].notnull())
        )
    ]
    bp[["diastolic", "systolic"]] = bp[["diastolic", "systolic"]].astype(int)
    bp.count()
    bp = bp[
        (bp["systolic"] >= 20)
        & (bp["systolic"] <= 300)
        & (bp["diastolic"] >= 5)
        & (bp["diastolic"] <= 200)
    ]
    # print(bp["diastolic"].mean(),bp["systolic"].mean())
    bp = bp.drop_duplicates(keep='last').reset_index(drop=True)
    bp = bp.groupby(['patid', 'indexdate', 'eventdate']).mean().reset_index()
    bp = bp.sort_values(["patid", "eventdate"]).reset_index(drop=True)

    # Date filtering removed; all records are kept.
    save_long_format_data(bp, save_long_format, 'bp')

    bp["bp_time_gap"] = (bp["indexdate"] - bp["eventdate"]).abs().dt.days
    bp = bp.loc[bp.groupby(["patid", "indexdate"])["bp_time_gap"].idxmin()].reset_index(drop=True)

    bp = bp[["patid", "eventdate", "indexdate", "systolic", "diastolic"]]
    bp = bp.rename(columns={"eventdate": "bp_date"})
    patient = patient.merge(bp, on=["patid", "indexdate"], how='left')
    return patient

# ------------------------------------------------------------------------------
# DATA LOADING & SUBSETTING
# ------------------------------------------------------------------------------
# Load Patient Data
patient = pd.read_csv(
    "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/baseline_with_all_features.txt",  # adjust filename/path accordingly
    sep="\t",
    dtype=str
)
patient['patid'] = patient['patid'].astype(str)
print(f"Patient data loaded: {len(patient)} rows.")

# --- CHANGED: ensure patient has indexdate (derive from earliest eventdate per patid) ---
if "indexdate" not in patient.columns:
    print("Note: 'indexdate' missing in patient — deriving from earliest eventdate per patid.")
    if "eventdate" in patient.columns:
        _tmp = patient[["patid", "eventdate"]].copy()
        _tmp["eventdate"] = pd.to_datetime(_tmp["eventdate"], errors="coerce", dayfirst=True)
        idx_map = (_tmp.dropna()
                        .groupby("patid", as_index=False)["eventdate"]
                        .min()
                        .rename(columns={"eventdate": "indexdate"}))
        patient = patient.merge(idx_map, on="patid", how="left")
    else:
        patient["indexdate"] = pd.NaT

# normalize types
patient["indexdate"] = pd.to_datetime(patient["indexdate"], errors="coerce", dayfirst=True)

# Load the Clinical Smoking Status File
clinical_smok = pd.read_csv(
    "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/Clinical_SmokingStatus_all.txt.gz",  # your gzipped clinical smoking file
    sep="\t",
    compression="gzip",
    header=0,
    parse_dates=['eventdate'],
    dayfirst=True
)
print(f"Clinical smoking file loaded: {len(clinical_smok)} rows.")

# --- CHANGED: fallback — derive indexdate from clinical_smok if still missing everywhere ---
if patient["indexdate"].isna().all():
    print("Deriving 'indexdate' from clinical_smok earliest eventdate per patid.")
    idx_map2 = (clinical_smok.groupby("patid", as_index=False)["eventdate"]
                .min()
                .rename(columns={"eventdate": "indexdate"}))
    patient = patient.drop(columns=["indexdate"], errors="ignore").merge(idx_map2, on="patid", how="left")
    patient["indexdate"] = pd.to_datetime(patient["indexdate"], errors="coerce", dayfirst=True)

# ------------------------------------------------------------------------------
# Locate ALL Additional Clinical Details files (now zipped) and stream in chunks
# ------------------------------------------------------------------------------

# Prefer .zip; fall back to .txt if any remain unzipped
add_zip_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Additional_*.zip"
add_txt_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Additional_*.txt"

add_files = sorted(glob.glob(add_zip_pattern))
if not add_files:
    add_files = sorted(glob.glob(add_txt_pattern))

print(f"Found {len(add_files)} additional clinical files.")
if not add_files:
    raise FileNotFoundError(f"No Additional files found for patterns:\n  {add_zip_pattern}\n  {add_txt_pattern}")

# Define temporary file names for each subset
temp_bp_file      = "temp_bp_data.csv"
temp_smoking_file = "temp_smoking_data.csv"
temp_weight_file  = "temp_weight_data.csv"
temp_height_file  = "temp_height_data.csv"

# Remove existing temporary files if they exist
for temp_file in [temp_bp_file, temp_smoking_file, temp_weight_file, temp_height_file]:
    if os.path.exists(temp_file):
        os.remove(temp_file)

# Only cols we rely on later; comment out usecols if your extract is missing any
# needed_cols = ["patid","enttype","eventdate","indexdate","data1","data2","data3"]

max_rows_limit = 20000  # Chunk size

for f in add_files:
    print(f"Processing file: {f}")
    compression = "zip" if f.lower().endswith(".zip") else "infer"
    reader = pd.read_csv(
        f,
        sep="\t",
        dtype=str,
        chunksize=max_rows_limit,
        compression=compression,
        # usecols=needed_cols,   # <- enable if all these columns exist
    )

    for chunk in reader:
        # make sure enttype is string
        if "enttype" in chunk.columns:
            chunk["enttype"] = chunk["enttype"].astype(str)
        else:
            raise KeyError("Column 'enttype' not found in Additional chunk.")

        # Filter and append to temp CSVs
        bp_chunk = chunk[chunk["enttype"] == "1"]
        if not bp_chunk.empty:
            bp_chunk.to_csv(temp_bp_file, mode='a', header=not os.path.exists(temp_bp_file), index=False)

        smoking_chunk = chunk[chunk["enttype"] == "4"]
        if not smoking_chunk.empty:
            smoking_chunk.to_csv(temp_smoking_file, mode='a', header=not os.path.exists(temp_smoking_file), index=False)

        weight_chunk = chunk[chunk["enttype"] == "13"]
        if not weight_chunk.empty:
            weight_chunk.to_csv(temp_weight_file, mode='a', header=not os.path.exists(temp_weight_file), index=False)

        height_chunk = chunk[chunk["enttype"] == "14"]
        if not height_chunk.empty:
            height_chunk.to_csv(temp_height_file, mode='a', header=not os.path.exists(temp_height_file), index=False)

# Load each subset from the temporary files (if they exist)
bp_data = pd.read_csv(temp_bp_file, dtype=str) if os.path.exists(temp_bp_file) else pd.DataFrame()
smoking_data = pd.read_csv(temp_smoking_file, dtype=str) if os.path.exists(temp_smoking_file) else pd.DataFrame()
weight_data = pd.read_csv(temp_weight_file, dtype=str) if os.path.exists(temp_weight_file) else pd.DataFrame()
height_data = pd.read_csv(temp_height_file, dtype=str) if os.path.exists(temp_height_file) else pd.DataFrame()

# Ensure required date columns exist and are in datetime format for each subset
if "indexdate" not in bp_data.columns:
    bp_data = bp_data.merge(patient[['patid', 'indexdate']], on='patid', how='left')
if "eventdate" not in bp_data.columns:
    bp_data["eventdate"] = bp_data["indexdate"]
bp_data["eventdate"] = pd.to_datetime(bp_data["eventdate"], errors="coerce", dayfirst=True)
bp_data["indexdate"] = pd.to_datetime(bp_data["indexdate"], errors="coerce", dayfirst=True)

if "indexdate" not in smoking_data.columns:
    smoking_data = smoking_data.merge(patient[['patid', 'indexdate']], on='patid', how='left')
if "eventdate" not in smoking_data.columns:
    smoking_data["eventdate"] = smoking_data["indexdate"]
smoking_data["eventdate"] = pd.to_datetime(smoking_data["eventdate"], errors="coerce", dayfirst=True)
smoking_data["indexdate"] = pd.to_datetime(smoking_data["indexdate"], errors="coerce", dayfirst=True)

if "indexdate" not in weight_data.columns:
    weight_data = weight_data.merge(patient[['patid', 'indexdate']], on='patid', how='left')
if "eventdate" not in weight_data.columns:
    weight_data["eventdate"] = weight_data["indexdate"]
weight_data["eventdate"] = pd.to_datetime(weight_data["eventdate"], errors="coerce", dayfirst=True)
weight_data["indexdate"] = pd.to_datetime(weight_data["indexdate"], errors="coerce", dayfirst=True)

if "indexdate" not in height_data.columns:
    height_data = height_data.merge(patient[['patid', 'indexdate']], on='patid', how='left')
if "eventdate" not in height_data.columns:
    height_data["eventdate"] = height_data["indexdate"]
height_data["eventdate"] = pd.to_datetime(height_data["eventdate"], errors="coerce", dayfirst=True)
height_data["indexdate"] = pd.to_datetime(height_data["indexdate"], errors="coerce", dayfirst=True)

# Load HES Hospital Data
hes_hosp = pd.read_csv(
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/hes_diagnosis_hosp_23_002869_DM.txt",  # adjust filename/path as necessary
    sep="\t",
    dtype=str
)
if "admidate" in hes_hosp.columns:
    hes_hosp["admidate"] = pd.to_datetime(hes_hosp["admidate"], errors="coerce", dayfirst=True)
hes_hosp['patid'] = hes_hosp['patid'].astype(str)
if "indexdate" not in hes_hosp.columns:
    print("indexdate missing from hes_hosp; merging from patient.")
    hes_hosp = hes_hosp.merge(patient[['patid', 'indexdate']], on='patid', how='left')
print(f"HES hospital data loaded: {len(hes_hosp)} rows.")
globals()['hes_hosp'] = hes_hosp

# ------------------------------------------------------------------------------
# DATA PROCESSING
# ------------------------------------------------------------------------------
# Process Smoking Data using smoking_data subset
cleaned_patient = get_smoking_data(smoking_data, clinical_smok, patient)
print("get_smoking_data processing complete.")

# Process BMI Data using concatenated weight and height data
wh_data = pd.concat([weight_data, height_data], ignore_index=True)
cleaned_patient = weight_height_bmi(wh_data, cleaned_patient)
print("weight_height_bmi processing complete.")

# Process Blood Pressure Data using bp_data subset
cleaned_patient = get_bp_data(bp_data, cleaned_patient)
print("get_bp_data processing complete.")

# ------------------------------------------------------------------------------
# Save the Cleaned Data (smoking, BMI and BP)
output_file = "Cleaned_Patient_Smoking_Data.csv"
cleaned_patient.to_csv(output_file, index=False)
print(f"Cleaned data saved to {output_file}")
