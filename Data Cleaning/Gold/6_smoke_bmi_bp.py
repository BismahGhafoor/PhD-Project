#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cleans smoking, BMI and blood pressure using ALL Additional Clinical files.

Outputs:
  - temp_bp_data.txt, temp_smoking_data.txt, temp_weight_data.txt, temp_height_data.txt  (tab-delimited)
  - Cleaned_Patient_Smoking_Data.txt  (tab-delimited)

Reads:
  - Clinical_SmokingStatus_all.txt.gz (gzipped TSV)
  - Additional files (.zip or .txt), streamed in chunks
  - baseline_with_all_features.txt (TSV)
  - HES diagnosis (TSV)
"""

import pandas as pd
import numpy as np
import glob
import os
import re

print("starting...")

# ----------------------------------------------------------------------
# helper functions you said you have
from helper_functions import lcf, ucf, perc
from helper_functions import save_long_format_data, read_long_format_data
from helper_functions import remap_eth, nperc_counts, calc_gfr

save_long_format = False  # make sure any helpers don’t silently write CSVs

# ----------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------
def get_smoking_data(smoking_data, clinical_smok, patient):
    # keys as string
    smoking_data['patid']  = smoking_data['patid'].astype(str)
    clinical_smok['patid'] = clinical_smok['patid'].astype(str)
    patient['patid']       = patient['patid'].astype(str)

    # --- Additional (enttype 4)
    print("Columns in smoking_data before subsetting:", smoking_data.columns.tolist())
    smok_add = smoking_data[['patid', "indexdate", 'eventdate', 'data1']].copy()
    smok_add["data1"] = pd.to_numeric(smok_add["data1"], errors="coerce")
    smok_add = smok_add[(smok_add['data1'].isin([1, 2, 3])) & (smok_add['eventdate'].notnull())]
    smok_add = smok_add.sort_values(['patid', 'eventdate']).reset_index(drop=True)
    smok_add = smok_add.sample(frac=1).drop_duplicates(subset=['patid', 'eventdate'], keep='last')
    smok_add['data1'] = smok_add['data1'].replace({1: "Yes", 2: "No", 3: "Ex"})
    nperc_counts(smok_add, 'data1')
    smok_add = smok_add.rename(columns={'data1': 'smok_add'})

    # --- Clinical (medcodes)
    smok_clref = clinical_smok.copy(deep=True)
    if "indexdate" not in smok_clref.columns:
        print("indexdate missing from clinical_smok; merging from patient.")
        smok_clref = smok_clref.merge(patient[['patid', 'indexdate']], on='patid', how='left')
    if "eventdate" not in smok_clref.columns:
        print("eventdate missing from clinical_smok; copying indexdate to eventdate.")
        smok_clref["eventdate"] = smok_clref["indexdate"]

    smok_clref['smok_clref'] = np.nan
    smoke_keys = ['current smoker', 'never smoker', 'ex smoker']
    smoke_cat  = ['Yes', "No", 'Ex']
    filepath   = '/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/GOLD_Codes_FZ.xlsx'
    smoke_codes = pd.read_excel(filepath, sheet_name="Smok")
    smoke_codes = smoke_codes[smoke_codes.source == 'cprd']
    smoke_values = [smoke_codes[smoke_codes['type'] == key]['medcode'].to_list() for key in smoke_keys]
    smoke_dict   = dict(zip(smoke_keys, smoke_values))

    for idx, key in enumerate(smoke_keys):
        smok_clref.loc[smok_clref['medcode'].isin(smoke_dict.get(key, [])), 'smok_clref'] = smoke_cat[idx]
    smok_clref = smok_clref[smok_clref['smok_clref'].notnull()]
    smok_clref = smok_clref.sort_values(['patid', 'eventdate']).reset_index(drop=True)
    smok_clref = smok_clref.sample(frac=1).drop_duplicates(subset=['patid', 'eventdate'], keep='last')
    smok_clref = smok_clref[['patid', "indexdate", 'eventdate', 'smok_clref']]

    # --- HES (ICD)
    hes_codes = pd.read_excel(filepath, sheet_name="Smok")
    hes_codes = hes_codes[hes_codes.source == 'hes'].medcode.tolist()
    if hes_codes:
        # build a regex that matches codes at the start (prefix match)
        pat = r'^(?:' + '|'.join(re.escape(c) for c in hes_codes) + r')'
        mask = hes_hosp["ICD"].fillna('').str.contains(pat, regex=True, na=False)
        smok_hes = hes_hosp[mask].copy()
    else:
        smok_hes = hes_hosp.iloc[0:0].copy()

    smok_hes = smok_hes.sort_values(['patid', 'admidate']).reset_index(drop=True)
    smok_hes = smok_hes.sample(frac=1).drop_duplicates(subset=['patid', 'admidate'], keep='last')
    smok_hes['smok_hes'] = 'Yes'
    smok_hes = smok_hes.rename(columns={'admidate': 'eventdate'})
    if "indexdate" not in smok_hes.columns:
        print("indexdate missing from smok_hes; merging from patient.")
        smok_hes = smok_hes.merge(patient[['patid', 'indexdate']], on='patid', how='left')
    smok_hes = smok_hes[['patid', "indexdate", 'eventdate', 'smok_hes']]

    # --- Merge preference: clinical → HES → additional
    smoking = smok_clref.merge(smok_add, how='outer', on=['patid', "indexdate", 'eventdate'])
    smoking = smoking.merge(smok_hes, how='outer', on=['patid', "indexdate", 'eventdate'])
    smoking['smoking_status'] = smoking['smok_clref']
    smoking.loc[smoking['smoking_status'].isna() & smoking['smok_hes'].notna(), 'smoking_status'] = smoking['smok_hes']
    smoking.loc[smoking['smoking_status'].isna() & smoking['smok_add'].notna(), 'smoking_status'] = smoking['smok_add']
    nperc_counts(smoking, 'smoking_status')
    smoking = smoking.sort_values(['patid', 'eventdate']).reset_index(drop=True)
    smoking = smoking.sample(frac=1).drop_duplicates(subset=['patid', 'eventdate'], keep='last')
    smoking = smoking[['patid', 'indexdate', 'eventdate', 'smoking_status']]
    save_long_format_data(smoking, save_long_format, 'smoking')
    smoking['smok_time_gap'] = (smoking['indexdate'] - smoking['eventdate']).abs().dt.days
    smoking = smoking.loc[smoking.groupby(['patid', 'indexdate'])['smok_time_gap'].idxmin()].reset_index(drop=True)
    smoking = smoking.rename(columns={"eventdate": "smoking_date"})
    patient = patient.merge(smoking[['patid','indexdate','smoking_date','smoking_status']], on=['patid', 'indexdate'], how='left')
    return patient

def weight_height_bmi(wh_data, patient):
    weight = wh_data[wh_data["enttype"] == "13"][["patid", "indexdate", "eventdate", "data1", "data3"]].rename(
        columns={"data1": "weight_kg", "data3": "bmi_recorded"}
    )
    height = wh_data[wh_data["enttype"] == "14"][["patid", "indexdate", "eventdate", "data1"]].rename(
        columns={"data1": "height_m"}
    )
    bmi = weight.merge(height, how="outer", on=["patid", "indexdate", "eventdate"])
    bmi[["weight_kg", "height_m", "bmi_recorded"]] = bmi[["weight_kg", "height_m", "bmi_recorded"]].apply(pd.to_numeric, errors="coerce")
    bmi["height_m"] = bmi.groupby(["patid", "indexdate"])["height_m"].ffill()
    bmi["bmi_calc"] = bmi["weight_kg"] / (bmi["height_m"] * bmi["height_m"])
    bmi["bmi"] = bmi["bmi_recorded"]
    bmi.loc[(bmi["bmi"].isna() | (bmi["bmi"].isin([0, np.inf]))) & np.isfinite(bmi["bmi_calc"]), "bmi"] = bmi["bmi_calc"]
    bmi = bmi[(bmi["bmi"] >= 10) & (bmi["bmi"] <= 70)]
    bmi = bmi.drop_duplicates(subset=['patid', 'indexdate', 'eventdate', 'bmi'], keep='last').reset_index(drop=True)
    bmi = bmi.groupby(['patid', 'indexdate', 'eventdate']).mean(numeric_only=True).reset_index()
    bmi = bmi.sort_values(["patid", "eventdate"]).reset_index(drop=True)
    save_long_format_data(bmi, save_long_format, 'bmi')
    bmi["bmi_time_gap"] = (bmi["indexdate"] - bmi["eventdate"]).abs().dt.days
    bmi = bmi.loc[bmi.groupby(["patid", "indexdate"])["bmi_time_gap"].idxmin()].reset_index(drop=True)
    bmi = bmi.rename(columns={"eventdate": "bmi_date"})
    patient = patient.merge(bmi[["patid","indexdate","bmi_date","bmi"]], on=["patid", "indexdate"], how="left")
    return patient

def get_bp_data(bp_data, patient):
    bp = bp_data[["patid", "eventdate", "indexdate", "data1", "data2"]].rename(columns={"data1": "diastolic", "data2": "systolic"}).copy()
    bp["diastolic"] = pd.to_numeric(bp["diastolic"], errors="coerce")
    bp["systolic"]  = pd.to_numeric(bp["systolic"],  errors="coerce")
    bp = bp[(bp["eventdate"].notnull()) & bp["diastolic"].notnull() & bp["systolic"].notnull()]
    bp = bp[(bp["systolic"] >= 20) & (bp["systolic"] <= 300) & (bp["diastolic"] >= 5) & (bp["diastolic"] <= 200)]
    bp = bp.drop_duplicates(keep='last').reset_index(drop=True)
    bp = bp.groupby(['patid', 'indexdate', 'eventdate']).mean(numeric_only=True).reset_index()
    bp = bp.sort_values(["patid", "eventdate"]).reset_index(drop=True)
    save_long_format_data(bp, save_long_format, 'bp')
    bp["bp_time_gap"] = (bp["indexdate"] - bp["eventdate"]).abs().dt.days
    bp = bp.loc[bp.groupby(["patid", "indexdate"])["bp_time_gap"].idxmin()].reset_index(drop=True)
    bp = bp.rename(columns={"eventdate": "bp_date"})
    patient = patient.merge(bp[["patid","indexdate","bp_date","systolic","diastolic"]], on=["patid", "indexdate"], how='left')
    return patient

def ensure_dates(df, patient):
    """Ensure patid is str, add/parse indexdate and eventdate; return updated df."""
    if df.empty:
        return df
    df = df.copy()
    df['patid'] = df['patid'].astype(str)
    if "indexdate" not in df.columns:
        df = df.merge(patient[['patid','indexdate']], on='patid', how='left')
    if "eventdate" not in df.columns:
        df["eventdate"] = df["indexdate"]
    df["eventdate"] = pd.to_datetime(df["eventdate"], errors="coerce", dayfirst=True)
    df["indexdate"] = pd.to_datetime(df["indexdate"], errors="coerce", dayfirst=True)
    return df

# ----------------------------------------------------------------------
# LOADS
# ----------------------------------------------------------------------
patient = pd.read_csv("/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/baseline_with_all_features.txt",
                      sep="\t", dtype=str)
patient['patid'] = patient['patid'].astype(str)
print(f"Patient data loaded: {len(patient)} rows.")

if "indexdate" not in patient.columns:
    print("Note: 'indexdate' missing — deriving from earliest eventdate per patid if available.")
    if "eventdate" in patient.columns:
        _tmp = patient[["patid", "eventdate"]].copy()
        _tmp["eventdate"] = pd.to_datetime(_tmp["eventdate"], errors="coerce", dayfirst=True)
        idx_map = (_tmp.dropna().groupby("patid", as_index=False)["eventdate"].min()
                   .rename(columns={"eventdate": "indexdate"}))
        patient = patient.merge(idx_map, on="patid", how="left")
    else:
        patient["indexdate"] = pd.NaT
patient["indexdate"] = pd.to_datetime(patient["indexdate"], errors="coerce", dayfirst=True)

clinical_smok = pd.read_csv("/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/Clinical_SmokingStatus_all.txt.gz",
                            sep="\t", compression="gzip", header=0,
                            parse_dates=['eventdate'], dayfirst=True)
clinical_smok['patid'] = clinical_smok['patid'].astype(str)  # <-- important
print(f"Clinical smoking file loaded: {len(clinical_smok)} rows.")

if patient["indexdate"].isna().all():
    print("Deriving 'indexdate' from clinical_smok earliest eventdate per patid.")
    idx_map2 = (clinical_smok.assign(patid=clinical_smok['patid'].astype(str))
                .groupby("patid", as_index=False)["eventdate"].min()
                .rename(columns={"eventdate": "indexdate"}))
    patient = (patient.drop(columns=["indexdate"], errors="ignore")
                      .merge(idx_map2, on="patid", how="left"))
    patient["indexdate"] = pd.to_datetime(patient["indexdate"], errors="coerce", dayfirst=True)

# ----------------------------------------------------------------------
# Stream Additional files; write TEMP **TXT** chunks
# ----------------------------------------------------------------------
add_zip_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Additional_*.zip"
add_txt_pattern = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Additional_*.txt"

add_files = sorted(glob.glob(add_zip_pattern)) or sorted(glob.glob(add_txt_pattern))
print(f"Found {len(add_files)} additional clinical files.")
if not add_files:
    raise FileNotFoundError(f"No Additional files found:\n  {add_zip_pattern}\n  {add_txt_pattern}")

# TEMP FILES (TXT)
temp_bp_file      = "temp_bp_data.txt"
temp_smoking_file = "temp_smoking_data.txt"
temp_weight_file  = "temp_weight_data.txt"
temp_height_file  = "temp_height_data.txt"

for temp_file in (temp_bp_file, temp_smoking_file, temp_weight_file, temp_height_file):
    if os.path.exists(temp_file):
        os.remove(temp_file)

max_rows_limit = 20000

for f in add_files:
    print(f"Processing file: {f}")
    compression = "zip" if f.lower().endswith(".zip") else "infer"
    reader = pd.read_csv(f, sep="\t", dtype=str, chunksize=max_rows_limit, compression=compression)

    for chunk in reader:
        if "enttype" not in chunk.columns:
            raise KeyError("Column 'enttype' not found in Additional chunk.")
        chunk["enttype"] = chunk["enttype"].astype(str)

        bp_chunk = chunk[chunk["enttype"] == "1"]
        if not bp_chunk.empty:
            bp_chunk.to_csv(temp_bp_file, sep="\t", mode='a',
                            header=not os.path.exists(temp_bp_file), index=False)

        smoking_chunk = chunk[chunk["enttype"] == "4"]
        if not smoking_chunk.empty:
            smoking_chunk.to_csv(temp_smoking_file, sep="\t", mode='a',
                                 header=not os.path.exists(temp_smoking_file), index=False)

        weight_chunk = chunk[chunk["enttype"] == "13"]
        if not weight_chunk.empty:
            weight_chunk.to_csv(temp_weight_file, sep="\t", mode='a',
                                header=not os.path.exists(temp_weight_file), index=False)

        height_chunk = chunk[chunk["enttype"] == "14"]
        if not height_chunk.empty:
            height_chunk.to_csv(temp_height_file, sep="\t", mode='a',
                                header=not os.path.exists(temp_height_file), index=False)

# Load subsets from TEMP **TXT**
bp_data      = pd.read_csv(temp_bp_file, sep="\t", dtype=str) if os.path.exists(temp_bp_file) else pd.DataFrame()
smoking_data = pd.read_csv(temp_smoking_file, sep="\t", dtype=str) if os.path.exists(temp_smoking_file) else pd.DataFrame()
weight_data  = pd.read_csv(temp_weight_file, sep="\t", dtype=str) if os.path.exists(temp_weight_file) else pd.DataFrame()
height_data  = pd.read_csv(temp_height_file, sep="\t", dtype=str) if os.path.exists(temp_height_file) else pd.DataFrame()

# Ensure required date columns (datetime) — assign back
bp_data      = ensure_dates(bp_data, patient)
smoking_data = ensure_dates(smoking_data, patient)
weight_data  = ensure_dates(weight_data, patient)
height_data  = ensure_dates(height_data, patient)

# ----------------------------------------------------------------------
# HES (ensure ICD present/standardized)
# ----------------------------------------------------------------------
hes_hosp = pd.read_csv(
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/hes_diagnosis_hosp_23_002869_DM.txt",
    sep="\t", dtype=str
)
if "admidate" in hes_hosp.columns:
    hes_hosp["admidate"] = pd.to_datetime(hes_hosp["admidate"], errors="coerce", dayfirst=True)
hes_hosp['patid'] = hes_hosp['patid'].astype(str)

icd_col = 'ICD' if 'ICD' in hes_hosp.columns else ('diag_icd10' if 'diag_icd10' in hes_hosp.columns else None)
if icd_col is None:
    raise KeyError("No ICD column found in HES file (expected 'ICD' or 'diag_icd10').")
hes_hosp['ICD'] = hes_hosp[icd_col].astype(str).str.strip().str.upper()

if "indexdate" not in hes_hosp.columns:
    print("indexdate missing from hes_hosp; merging from patient.")
    hes_hosp = hes_hosp.merge(patient[['patid', 'indexdate']], on='patid', how='left')

globals()['hes_hosp'] = hes_hosp  # used in get_smoking_data

# ----------------------------------------------------------------------
# PROCESS
# ----------------------------------------------------------------------
cleaned_patient = get_smoking_data(smoking_data, clinical_smok, patient)
print("get_smoking_data complete.")

wh_data = pd.concat([weight_data, height_data], ignore_index=True)
cleaned_patient = weight_height_bmi(wh_data, cleaned_patient)
print("weight_height_bmi complete.")

cleaned_patient = get_bp_data(bp_data, cleaned_patient)
print("get_bp_data complete.")

# ----------------------------------------------------------------------
# SAVE (TXT)
# ----------------------------------------------------------------------
output_file = "Cleaned_Patient_Smoking_Data.txt"
cleaned_patient.to_csv(output_file, sep="\t", index=False)
print(f"Cleaned data saved to {output_file}")
