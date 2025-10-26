#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract serum cholesterol, LDL, HDL, triglycerides, and HbA1c
from your patient and test data files without applying any date filters.
Final output is a tab-delimited TXT.

Inputs
- /scratch/alice/b/bg205/DataCleaning_FINAL_Gold/Cleaned_Patient_Smoking_Data.txt  (TSV)
- Test_entities_all.txt.gz  (gzipped TSV with patid, enttype, value, unit, eventdate)
- /rfs/LRWE_Proj88/.../GOLD/TXTFILES/SUM.txt  (unit lookup)

Output
- extracted_lab_data.txt  (TSV)
"""

import os
import pandas as pd
import numpy as np
import tempfile

print("Starting lab data extraction (TSV)…")

# ------------------------------------------------------------------------------
# Helper imports (no changes to your helpers)
# ------------------------------------------------------------------------------
from helper_functions import lcf, ucf, perc
from helper_functions import save_long_format_data, read_long_format_data
from helper_functions import remap_eth, nperc_counts, calc_gfr

save_long_format = False

# ------------------------------------------------------------------------------
# Temp dir (optional)
# ------------------------------------------------------------------------------
temp_dir = tempfile.TemporaryDirectory()
print("Temporary directory:", temp_dir.name)

# ------------------------------------------------------------------------------
# Load unit lookups (once) and prepare allowed codes
# ------------------------------------------------------------------------------
UNITS_PATH = "/rfs/LRWE_Proj88/Shared/Cohort_Definition/Denominator_Linkage_CPRD_Cohort_Data/GOLD_Lookups/GOLD/TXTFILES/SUM.txt"
units_df = pd.read_csv(UNITS_PATH, sep="\t", header=0)
units_df["Code"] = pd.to_numeric(units_df["Code"], errors="coerce").astype("Int64")
unit_name_col = "Specimen Unit of Measure" if "Specimen Unit of Measure" in units_df.columns else units_df.columns[-1]
# Accept any code whose unit name contains mmol/L (robust to case/spacing)
_name = units_df[unit_name_col].astype(str).str.lower()
lipid_mmol_codes = set(units_df.loc[_name.str.contains("mmol") & _name.str.contains("/l"), "Code"].dropna().astype(int).tolist())

# HbA1c code sets used in conversions
hba1c_pct_codes   = {1, 215}
hba1c_mmolmol_codes = {97, 156, 205, 187}
hba1c_dcct96_codes  = {96}

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _detect_measurement_and_code(df_sub: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Heuristically determine which column holds the measurement and which holds the unit code.
    Works with files where 'value' and 'unit' are swapped (as in your sample).
    Returns (measurement_series_float, unit_code_series_Int64).
    """
    v = pd.to_numeric(df_sub["value"], errors="coerce")
    u = pd.to_numeric(df_sub["unit"],  errors="coerce")

    # Fraction that are fractional numbers (not integers)
    frac_v = (v % 1 != 0).mean(skipna=True)
    frac_u = (u % 1 != 0).mean(skipna=True)

    # Heuristic: for most labs, the measurement has more fractional values than the code column.
    # Your sample clearly has measurement in 'unit' (e.g., 6.44) and code in 'value' (e.g., 3).
    if frac_u > frac_v:
        meas = u
        code = v.round(0).astype("Int64")
        src = "unit->meas, value->code"
    else:
        meas = v
        code = u.round(0).astype("Int64")
        src = "value->meas, unit->code"

    print(f"[detect] chose {src}; frac_v={frac_v:.2f}, frac_u={frac_u:.2f}")
    return meas, code


def _merge_nearest_to_indexdate(patient: pd.DataFrame,
                                df: pd.DataFrame,
                                value_col: str,
                                date_col_name: str) -> pd.DataFrame:
    df["time_gap"] = (df["indexdate"] - df["eventdate"]).abs().dt.days
    df = df.sort_values(["patid", "indexdate", "time_gap"])
    df = df.loc[df.groupby(["patid", "indexdate"])["time_gap"].idxmin()].reset_index(drop=True)
    df = df.rename(columns={"eventdate": date_col_name})
    out = patient.merge(df[["patid", "indexdate", date_col_name, value_col]], on=["patid", "indexdate"], how="left")
    return out

# ------------------------------------------------------------------------------
# Lab extractors
# ------------------------------------------------------------------------------
def get_lab_data(test: pd.DataFrame,
                 patient: pd.DataFrame,
                 enttype: str,
                 unit_code: float,
                 limit: float,
                 col: str) -> pd.DataFrame:
    """
    Filter test by enttype & unit (accept any mmol/L code), keep values <= limit,
    pick record closest to indexdate, merge back to patient.
    """
    enttype_str = str(enttype)
    print(f"\n[get_lab_data] {col}: enttype={enttype_str}, requested_unit_code={unit_code}, limit={limit}")

    df = test[test["enttype"] == enttype_str].copy()
    if df.empty:
        print("[get_lab_data] No rows for this enttype.")
        return patient

    # Ensure dates exist or map from patient
    if "indexdate" not in df.columns:
        df["indexdate"] = df["patid"].map(patient.set_index("patid")["indexdate"])

    df["eventdate"] = pd.to_datetime(df["eventdate"], errors="coerce", dayfirst=True)
    df["indexdate"] = pd.to_datetime(df["indexdate"], errors="coerce", dayfirst=True)

    # Detect which column is measurement vs. unit code
    meas, code = _detect_measurement_and_code(df)

    # Diagnostics
    print(f"[get_lab_data] sample meas describe: count={meas.count():,}, median={meas.median(skipna=True):.3f}")
    top_codes = code.value_counts().head(10).to_dict()
    print(f"[get_lab_data] top unit codes: {top_codes}")

    # Filter plausible rows
    df[col] = meas
    df["unit_code_int"] = code

    # Accept either the requested code (e.g., 96) OR anything that maps to mmol/L in SUM.txt
    before = len(df)
    code_match = df["unit_code_int"].isin(lipid_mmol_codes) | (df["unit_code_int"] == int(round(float(unit_code))))
    df = df[code_match & (df[col] <= float(limit))]
    print(f"[get_lab_data] After unit/value filter: kept {len(df)} / {before}")
    if df.empty:
        return patient

    # Keep necessary columns and save long format if desired
    df = df[["patid", "eventdate", "indexdate", col]].dropna(subset=["eventdate", "indexdate", col])
    save_long_format_data(df.copy(), save_long_format, col)

    # Merge nearest to indexdate
    out = _merge_nearest_to_indexdate(patient, df, value_col=col, date_col_name=f"{col}_date")
    print(f"[get_lab_data] Merged: patient rows={len(out)}")
    return out


def get_hba1c_data(patient: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """
    Extract HbA1c (enttype 275). Convert to percent using unit codes.
    Keep plausible 2–20%, choose nearest to indexdate, merge back.
    Handles swapped value/unit columns.
    """
    print("\n[get_hba1c_data] enttype=275")
    h = test[test["enttype"] == "275"].copy()
    if h.empty:
        print("[get_hba1c_data] No HbA1c rows.")
        return patient

    # Ensure dates exist or map from patient
    if "indexdate" not in h.columns:
        h["indexdate"] = h["patid"].map(patient.set_index("patid")["indexdate"])

    h["eventdate"] = pd.to_datetime(h["eventdate"], errors="coerce", dayfirst=True)
    h["indexdate"] = pd.to_datetime(h["indexdate"], errors="coerce", dayfirst=True)

    # Detect measurement vs. unit code
    meas, code = _detect_measurement_and_code(h)
    h["value_num"] = pd.to_numeric(meas, errors="coerce")
    h["unit_int"]  = pd.to_numeric(code, errors="coerce").round(0).astype("Int64")

    # Optional diagnostics
    print("[get_hba1c_data] top unit codes:", h["unit_int"].value_counts().head(10).to_dict())

    # Convert to %
    h["hba1c_perc"] = np.nan
    h.loc[h["unit_int"].isin(hba1c_pct_codes),       "hba1c_perc"] = h["value_num"]
    h.loc[h["unit_int"].isin(hba1c_mmolmol_codes),   "hba1c_perc"] = h["value_num"] * 0.0915 + 2.15
    h.loc[h["unit_int"].isin(hba1c_dcct96_codes),    "hba1c_perc"] = h["value_num"] * 0.6277 + 1.627

    # Plausible range
    before = len(h)
    h = h[(h["hba1c_perc"] > 2.0) & (h["hba1c_perc"] <= 20.0)].copy()
    print(f"[get_hba1c_data] After plausibility filter: kept {len(h)} / {before}")
    if h.empty:
        return patient

    # Keep necessary cols and save long
    h = h[["patid", "eventdate", "indexdate", "hba1c_perc"]].dropna(subset=["eventdate", "indexdate", "hba1c_perc"])
    save_long_format_data(h.copy(), save_long_format, "hba1c")

    # Merge nearest to indexdate
    out = _merge_nearest_to_indexdate(patient, h, value_col="hba1c_perc", date_col_name="hba1c_date")
    print(f"[get_hba1c_data] Merged: patient rows={len(out)}")
    return out

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Patient TSV (don’t parse dates on read; coerce after)
    print("\n[Main] Reading patient TSV…")
    patient = pd.read_csv(
        "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/Cleaned_Patient_Smoking_Data.txt",
        sep="\t", dtype=str
    )
    # Ensure patid is string
    patient["patid"] = patient["patid"].astype(str)

    # Coerce any date-like columns that exist
    for c in ["eventdate", "indexdate", "dod", "smoking_date", "bmi_date", "bp_date"]:
        if c in patient.columns:
            patient[c] = pd.to_datetime(patient[c], errors="coerce", dayfirst=True)

    print(f"[Main] Patient rows: {len(patient):,}")

    # IDs to keep in test
    keep_ids = set(patient["patid"].unique())

    # Read test gzipped TSV in chunks (filter to patient ids)
    print("\n[Main] Reading Test_entities_all.txt.gz in chunks…")
    # If your gz actually contains an indexdate column, it's fine to read it; not required though
    usecols = ["patid", "enttype", "value", "unit", "eventdate"]  # 'indexdate' not required; we map it
    chunks = []
    for i, ch in enumerate(pd.read_csv(
        "Test_entities_all.txt.gz",
        sep="\t", compression="gzip", header=0,
        usecols=usecols, parse_dates=["eventdate"], dayfirst=True,
        chunksize=100_000
    ), start=1):
        ch["patid"] = ch["patid"].astype(str)
        filt = ch[ch["patid"].isin(keep_ids)]
        chunks.append(filt)
        print(f"  chunk {i}: in={len(ch):,} kept={len(filt):,}")

    test = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols)
    print(f"[Main] Test combined: {test.shape}")

    # Normalize basic types in test
    if not test.empty:
        test["enttype"] = test["enttype"].astype(str)
        # We'll detect measurement vs code later per-enttype; still coerce for safety
        test["value"] = pd.to_numeric(test["value"], errors="coerce")
        test["unit"]  = pd.to_numeric(test["unit"],  errors="coerce")

        # Map indexdate from patient (keeps datetime dtype)
        test["indexdate"] = test["patid"].map(patient.set_index("patid")["indexdate"])

        # Quick sanity prints
        print("[Main] dtypes:", test[["enttype", "unit", "value"]].dtypes.to_dict())
        try:
            print("[DEBUG] enttype unique (top 5):", test["enttype"].value_counts().head(5).to_dict())
        except Exception:
            pass
        try:
            # For tot_chol, this will often show measurement values when file is swapped
            print("[DEBUG] enttype=163, 'unit' sample (top 5):",
                  test[test["enttype"] == "163"]["unit"].astype(str).value_counts().head(5).to_dict())
            print("[DEBUG] enttype=163, 'value' sample (top 5):",
                  test[test["enttype"] == "163"]["value"].astype(str).value_counts().head(5).to_dict())
        except Exception:
            pass

    # Extract labs (unit_code kept for backward compatibility; filter also accepts any mmol/L code)
    patient = get_lab_data(test, patient, enttype="163", unit_code=96.0, limit=20, col="tot_chol")
    patient = get_lab_data(test, patient, enttype="175", unit_code=96.0, limit=20, col="hdl")
    patient = get_lab_data(test, patient, enttype="177", unit_code=96.0, limit=20, col="ldl")
    patient = get_lab_data(test, patient, enttype="202", unit_code=96.0, limit=20, col="trigly")
    patient = get_hba1c_data(patient, test)

    # Save TSV
    final_path = os.path.join(os.getcwd(), "extracted_lab_data.txt")
    patient.to_csv(final_path, sep="\t", index=False, date_format="%d/%m/%Y")
    print(f"\n[Main] Saved: {final_path}")

    # Preview
    print(patient.head())
