#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract serum total cholesterol, LDL, HDL, triglycerides, and HbA1c from GOLD test data.
TXT-only I/O where possible.

Reads:
  - /scratch/.../Cleaned_Patient_Smoking_Data.txt (TSV)
  - Test_entities_all.txt.gz                      (gzipped TSV)
  - SUM.txt (unit lookup)                         (TSV)

Writes:
  - extracted_lab_data.txt                        (TSV)
"""

import os
import pandas as pd
import numpy as np

print("Starting lab data extraction…")

# ---- your helpers (present in your env) ----
from helper_functions import lcf, ucf, perc
from helper_functions import save_long_format_data, read_long_format_data
from helper_functions import remap_eth, nperc_counts, calc_gfr

save_long_format = False

# ---------------- core helpers ----------------
def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def get_lab_data(test, patient, enttype, unit_code, limit, col):
    """
    Pull a single lab by enttype, keep rows with unit==unit_code (mmol/L stream),
    numeric value <= limit, pick record closest to indexdate, then merge into patient.
    """
    enttype_str = str(enttype)
    print(f"\n[get_lab_data] {col}: enttype={enttype_str}, unit={unit_code}, limit={limit}")

    df = test[test["enttype"] == enttype_str].copy()
    if df.empty:
        print(f"[get_lab_data] No rows for enttype={enttype_str}")
        return patient

    df["value"] = _to_num(df["value"])
    df["unit_str"] = df["unit"].astype(str)
    unit_code_str = str(int(unit_code)) if isinstance(unit_code, (int, float)) and not pd.isna(unit_code) else str(unit_code)

    df = df[(df["unit_str"] == unit_code_str) & df["value"].notna() & (df["value"] <= float(limit))]
    print(f"[get_lab_data] After unit/value filter: {len(df)} rows")
    if df.empty:
        return patient

    # ensure indexdate available (should already be merged into test; keep safe)
    if "indexdate" not in df.columns:
        df = df.merge(patient[["patid", "indexdate"]], on="patid", how="left")

    df["time_gap"] = (df["indexdate"] - df["eventdate"]).abs().dt.days
    df = df.loc[df.groupby(["patid", "indexdate"])["time_gap"].idxmin()].copy()

    df = df.rename(columns={"eventdate": f"{col}_date", "value": col})
    keep = ["patid", "indexdate", f"{col}_date", col]
    out = patient.merge(df[keep], on=["patid", "indexdate"], how="left")
    print(f"[get_lab_data] Merged {col}: patient shape = {out.shape}")
    return out

def get_hba1c_data(patient, test):
    """
    HbA1c from enttype=275.
    Convert to %:
      - % codes {1,215}: keep
      - IFCC mmol/mol {97,156,205,187}: % = 0.0915*mmol/mol + 2.15
      - special {96}: % = 0.6277*value + 1.627
    Keep 2–20%, pick closest to indexdate, merge.
    """
    print("\n[get_hba1c_data] enttype=275")
    h = test[test["enttype"] == "275"].copy()
    if h.empty:
        print("[get_hba1c_data] No rows for enttype=275")
        return patient

    h["value"] = _to_num(h["value"])
    h["unit_num"] = _to_num(h["unit"])

    # Optional: map code->name (for logs only)
    try:
        units = pd.read_csv(
            "/rfs/LRWE_Proj88/Shared/Cohort_Definition/Denominator_Linkage_CPRD_Cohort_Data/GOLD_Lookups/GOLD/TXTFILES/SUM.txt",
            sep="\t", header=0, dtype={"Code": float}
        )
        units["Code"] = _to_num(units["Code"])
        unit_name_map = units.set_index("Code")["Specimen Unit of Measure"]
        h["unit_name"] = h["unit_num"].map(unit_name_map)
        print("[get_hba1c_data] Unit counts (top 5):")
        print(h.groupby(["unit_num", "unit_name"]).size().sort_values(ascending=False).head())
    except Exception as e:
        print(f"[get_hba1c_data] Could not read SUM.txt ({e}); continuing without names.")
        h["unit_name"] = pd.NA

    pct_codes   = {1, 215}
    ifcc_codes  = {97, 156, 205, 187}
    special_96  = {96}

    h["hba1c_perc"] = np.nan
    h.loc[h["unit_num"].isin(pct_codes),  "hba1c_perc"] = h["value"]
    h.loc[h["unit_num"].isin(ifcc_codes), "hba1c_perc"] = 0.0915 * h["value"] + 2.15
    h.loc[h["unit_num"].isin(special_96), "hba1c_perc"] = 0.6277 * h["value"] + 1.627

    h = h[h["hba1c_perc"].between(2.0, 20.0, inclusive="both")].copy()
    if h.empty:
        print("[get_hba1c_data] No plausible % values after conversion")
        return patient

    if "indexdate" not in h.columns:
        h = h.merge(patient[["patid", "indexdate"]], on="patid", how="left")

    h = (h[["patid", "eventdate", "indexdate", "hba1c_perc"]]
           .drop_duplicates()
           .groupby(["patid", "indexdate", "eventdate"], as_index=False)
           .mean(numeric_only=True))

    h["gap"] = (h["indexdate"] - h["eventdate"]).abs().dt.days
    h = h.loc[h.groupby(["patid", "indexdate"])["gap"].idxmin()].copy()

    h = h.rename(columns={"eventdate": "hba1c_date"})
    out = patient.merge(h[["patid", "indexdate", "hba1c_date", "hba1c_perc"]],
                        on=["patid", "indexdate"], how="left")
    print(f"[get_hba1c_data] Merged HbA1c: patient shape = {out.shape}")
    return out

# ---------------- main ----------------
if __name__ == "__main__":
    # 1) Patient TSV (don’t assume 'eventdate' exists)
    patient_path = "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/Cleaned_Patient_Smoking_Data.txt"
    patient = pd.read_csv(patient_path, sep="\t", dtype=str)
    patient["patid"] = patient["patid"].astype(str)

    # Parse only date columns that actually exist
    possible_date_cols = ["indexdate", "dod", "smoking_date", "bmi_date", "bp_date"]
    for c in possible_date_cols:
        if c in patient.columns:
            patient[c] = pd.to_datetime(patient[c], errors="coerce", dayfirst=True)

    if "indexdate" not in patient.columns:
        # fallback: blank; later we map indexdate into test anyway
        patient["indexdate"] = pd.NaT

    print(f"[Main] Patient loaded: {patient.shape}")

    # 2) Test gz (filter to cohort patids)
    print("[Main] Loading Test_entities_all.txt.gz (filtered to cohort patids)…")
    cohort_ids = set(patient["patid"].unique())
    usecols = ["patid", "enttype", "value", "unit", "eventdate"]
    chunks = []
    for i, ch in enumerate(pd.read_csv(
        "Test_entities_all.txt.gz",
        sep="\t",
        compression="gzip",
        header=0,
        usecols=usecols,
        dtype=str,
        parse_dates=["eventdate"],
        dayfirst=True,
        chunksize=100_000
    ), 1):
        ch["patid"] = ch["patid"].astype(str)
        ch = ch[ch["patid"].isin(cohort_ids)]
        chunks.append(ch)
        print(f"  chunk {i}: kept {len(ch)} rows")

    test = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    print(f"[Main] Test combined: {test.shape}")

    # Ensure string enttype + add indexdate from patient
    test["enttype"] = test["enttype"].astype(str)
    # map indexdate (as datetime) onto test
    idx_map = patient[["patid", "indexdate"]].copy()
    test = test.merge(idx_map, on="patid", how="left")

    # 3) Lipids (unit 96 == mmol/L), cap 20 mmol/L
    patient = get_lab_data(test, patient, enttype="163", unit_code=96, limit=20, col="tot_chol")
    patient = get_lab_data(test, patient, enttype="175", unit_code=96, limit=20, col="hdl")
    patient = get_lab_data(test, patient, enttype="177", unit_code=96, limit=20, col="ldl")
    patient = get_lab_data(test, patient, enttype="202", unit_code=96, limit=20, col="trigly")

    # 4) HbA1c
    patient = get_hba1c_data(patient, test)

    # 5) Save TXT
    out_path = os.path.join(os.getcwd(), "extracted_lab_data.txt")
    patient.to_csv(out_path, sep="\t", index=False, date_format="%d/%m/%Y")
    print(f"\n[Main] Saved: {out_path}")

    # quick peek
    print(patient.head())
