#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract serum cholesterol, LDL, HDL, triglycerides, and HbA1c 
from your patient and test data files without applying any date filters.
Intermediate files are stored in a temporary directory, but the final output
is saved to the working directory.
"""

import os
import pandas as pd
import numpy as np
import tempfile

print("Starting lab data extraction script for troubleshooting...")

# ------------------------------------------------------------------------------
# Import helper functions used by get_smoking_data and long format saving
# ------------------------------------------------------------------------------
from helper_functions import lcf, ucf, perc
from helper_functions import save_long_format_data, read_long_format_data
from helper_functions import remap_eth, nperc_counts, calc_gfr

# Define flag to control saving of long format data (adjust as needed)
save_long_format = False

# ------------------------------------------------------------------------------
# Create a temporary directory for storing intermediate files
# ------------------------------------------------------------------------------
temp_dir = tempfile.TemporaryDirectory()
print("Temporary directory created at:", temp_dir.name)

# ------------------------------------------------------------------------------
# Function: get_lab_data
# ------------------------------------------------------------------------------
def get_lab_data(test, patient, enttype, unit, limit, col, save_long_format):
    """
    Extracts a specific lab measurement from the test dataframe,
    filters by unit and limit, selects the record closest to the patient's 
    index date, renames the event date column, and then merges the result with
    the patient dataframe.
    """
    # Use string comparison for enttype
    enttype_str = str(enttype)
    print(f"\n[get_lab_data] Processing {col}: enttype={enttype_str}, unit={unit}, limit={limit}")
    df = test[test["enttype"] == enttype_str].copy()
    print(f"[get_lab_data] After filtering by enttype: shape={df.shape}")
    
    df[col] = df["value"]
    df = df[(df["unit"] == unit) & (df[col] <= limit)]
    print(f"[get_lab_data] After filtering by unit and limit: shape={df.shape}")
    
    # Ensure that 'indexdate' exists in df. Map it from patient if necessary.
    if "indexdate" not in df.columns:
        df["indexdate"] = df["patid"].map(patient.set_index("patid")["indexdate"])
        print("[get_lab_data] 'indexdate' column added to test data by mapping from patient.")
    
    df = df[["patid", "eventdate", "indexdate"] + [col]]
    df = df.sort_values(["patid", "eventdate"]).reset_index(drop=True)
    print(f"[get_lab_data] After sorting: shape={df.shape}")
    
    # Date filters removed
    save_long_format_data(df, save_long_format, col)
    print(f"[get_lab_data] Long format data for {col} saved (if flag enabled).")
    
    df["time_gap"] = (df["indexdate"] - df["eventdate"]).abs().dt.days
    df = df.loc[df.groupby(["patid"])["time_gap"].idxmin()].reset_index(drop=True)
    print(f"[get_lab_data] After selecting record with min time_gap: shape={df.shape}")
    
    df = df[["patid", "eventdate", "indexdate"] + [col]]
    df = df.rename(columns={"eventdate": f"{col}_date"})
    patient = patient.merge(df, on=["patid", "indexdate"], how="left")
    print(f"[get_lab_data] After merging lab data for {col}: patient shape={patient.shape}")
    return patient

# ------------------------------------------------------------------------------
# Function: get_hba1c_data
# ------------------------------------------------------------------------------
def get_hba1c_data(patient, test):
    """
    Extracts HbA1c data from the test dataframe using enttype 275.
    Converts the measurement to a percentage using unit‐specific formulas,
    removes outlier values, selects the record closest to the index date,
    renames the event date column, and merges the result with the patient dataframe.
    """
    # Use string comparison for enttype
    print("\n[get_hba1c_data] Processing HbA1c data (enttype=275)")
    hba1c = test[test["enttype"] == "275"].copy()
    print(f"[get_hba1c_data] After filtering by enttype: shape={hba1c.shape}")
    
    # Read unit conversion file (adjust path as needed)
    units = pd.read_csv("/rfs/LRWE_Proj88/Shared/Cohort_Definition/Denominator_Linkage_CPRD_Cohort_Data/GOLD_Lookups/GOLD/TXTFILES/SUM.txt", 
                        sep="\t", header=0)
    hba1c["unit_name"] = hba1c["unit"].map(units.set_index("Code")["Specimen Unit of Measure"])
    print(f"[get_hba1c_data] Unique unit names: {hba1c['unit_name'].unique()}")
    
    grp = hba1c.groupby(["unit", "unit_name"]).size().reset_index(name="count")
    print(f"[get_hba1c_data] Group counts by unit (top 5): \n{grp.sort_values('count', ascending=False).head()}")
    
    # Convert to HbA1c percentage
    hba1c["hba1c_perc"] = np.nan
    hba1c["hba1c_perc"] = np.where(hba1c["unit"].isin([1, 215]), hba1c["value"], hba1c["hba1c_perc"])
    hba1c["hba1c_perc"] = np.where(hba1c["unit"].isin([97, 156, 205, 187]),
                                   hba1c["value"] * 0.0915 + 2.15,
                                   hba1c["hba1c_perc"])
    hba1c["hba1c_perc"] = np.where(hba1c["unit"].isin([96]),
                                   hba1c["value"] * 0.6277 + 1.627,
                                   hba1c["hba1c_perc"])
    print(f"[get_hba1c_data] Mean HbA1c after conversion: {hba1c['hba1c_perc'].mean():.2f}")
    
    # Remove outliers and average duplicates
    hba1c = hba1c[(hba1c["hba1c_perc"] > 2.0) & (hba1c["hba1c_perc"] <= 20)].reset_index(drop=True)
    print(f"[get_hba1c_data] After removing outliers: shape={hba1c.shape}")
    
    hba1c = hba1c[['patid', 'eventdate', 'indexdate', 'hba1c_perc']]
    hba1c = hba1c.sort_values(["patid", "eventdate"])
    hba1c = hba1c.drop_duplicates(keep='first')
    hba1c = hba1c.groupby(['patid', 'indexdate', 'eventdate']).mean().reset_index()
    print(f"[get_hba1c_data] After grouping duplicates: shape={hba1c.shape}")
    
    # Date filters removed
    save_long_format_data(hba1c, save_long_format, 'hba1c')
    print("[get_hba1c_data] Long format data for HbA1c saved (if flag enabled).")
    
    hba1c["hba1c_time_gap"] = (hba1c["eventdate"] - hba1c["indexdate"]).dt.days
    hba1c = hba1c.loc[hba1c.groupby(["patid", "indexdate"])["hba1c_time_gap"].apply(lambda x: x.abs().idxmin())].reset_index(drop=True)
    print(f"[get_hba1c_data] After selecting closest record: shape={hba1c.shape}")
    print("Mean HbA1c percentage:", hba1c["hba1c_perc"].mean())
    
    hba1c = hba1c[["patid", "eventdate", "indexdate", "hba1c_perc"]]
    hba1c = hba1c.rename(columns={"eventdate": "hba1c_date"})
    patient = patient.merge(hba1c, on=["patid", "indexdate"], how="left")
    print(f"[get_hba1c_data] After merging HbA1c data: patient shape={patient.shape}")
    return patient

# ------------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Read the patient data file (adjust file path and parameters as needed)
    print("\n[Main] Reading patient data...")
    patient = pd.read_csv(
        "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/Cleaned_Patient_Smoking_Data.csv",
        sep=",",
        header=0,
        parse_dates=["eventdate", "indexdate", "dod", "smoking_date", "bmi_date", "bp_date"],
        dayfirst=True
    )
    print(f"[Main] Patient data loaded: shape={patient.shape}")
    
    # Get the set of patient IDs for filtering
    patient_ids = set(patient["patid"].unique())
    print(f"[Main] Number of patients: {len(patient_ids)}")
    
    # ------------------------------------------------------------------------------
    # Read the test data file in chunks to manage memory usage and filter by patient IDs
    # ------------------------------------------------------------------------------
    print("\n[Main] Reading test data in chunks and filtering by patient IDs...")
    test_chunks = []
    chunksize = 100000
    usecols = ["patid", "enttype", "value", "unit", "eventdate"]
    chunk_num = 0
    for chunk in pd.read_csv(
            "Test_entities_all.txt.gz",
            sep="\t",
            compression="gzip",
            skipinitialspace=True,
            header=0,
            parse_dates=["eventdate"],
            dayfirst=True,
            chunksize=chunksize,
            usecols=usecols):
        chunk_num += 1
        # Filter chunk to only include rows with patid in patient_ids
        filtered_chunk = chunk[chunk["patid"].isin(patient_ids)]
        test_chunks.append(filtered_chunk)
        print(f"[Main] Processed chunk {chunk_num}: original shape={chunk.shape}, filtered shape={filtered_chunk.shape}")
    test = pd.concat(test_chunks, ignore_index=True)
    print(f"[Main] Test data loaded (combined & filtered): shape={test.shape}")
    
    # Convert enttype column to string so comparisons work as expected
    test["enttype"] = test["enttype"].astype(str)
    print(f"[Main] Converted 'enttype' to string. Unique enttype values: {test['enttype'].unique()}")
    
    # Map patient's indexdate to test data (if not present)
    test["indexdate"] = test["patid"].map(patient.set_index("patid")["indexdate"])
    print(f"[Main] Added 'indexdate' to test data. Unique indexdates: {test['indexdate'].nunique()}")
    
    # ------------------------------------------------------------------------------
    # Extract lab data
    # ------------------------------------------------------------------------------
    print("\n[Main] Extracting serum cholesterol (tot_chol)...")
    patient = get_lab_data(test, patient, enttype="163", unit=96.0, limit=20,
                            col="tot_chol", save_long_format=save_long_format)
    
    print("\n[Main] Extracting HDL...")
    patient = get_lab_data(test, patient, enttype="175", unit=96.0, limit=20,
                            col="hdl", save_long_format=save_long_format)
    
    print("\n[Main] Extracting LDL...")
    patient = get_lab_data(test, patient, enttype="177", unit=96.0, limit=20,
                            col="ldl", save_long_format=save_long_format)
    
    print("\n[Main] Extracting triglycerides (trigly)...")
    patient = get_lab_data(test, patient, enttype="202", unit=96.0, limit=20,
                            col="trigly", save_long_format=save_long_format)
    
    print("\n[Main] Extracting HbA1c...")
    patient = get_hba1c_data(patient, test)
    
    # Save the resulting patient dataframe with extracted lab data to a CSV file in the working directory.
    final_path = os.path.join(os.getcwd(), "extracted_lab_data.csv")
    patient.to_csv(final_path, index=False, sep=",", date_format='%d/%m/%Y')
    print(f"\n[Main] Final extracted lab data saved to working directory: {final_path}")
    
    # Print a preview of the resulting patient dataframe for troubleshooting.
    print("\n[Main] Preview of extracted lab data:")
    print(patient.head())
    
    # Optionally, clean up the temporary directory if desired.
    # temp_dir.cleanup()
