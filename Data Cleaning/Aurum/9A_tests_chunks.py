#!/usr/bin/env python3
# Extract HDL, LDL, Triglycerides, HbA1c (from Excel tabs) and Total Cholesterol (from TXT)
# from CPRD Aurum Observation ZIPs into per-ZIP gzipped CSV chunks.

import os
import sys
import zipfile
import pandas as pd

# ==============================
# CONFIG — EDIT THESE PATHS ONLY
# ==============================
# Folder containing the Aurum Observation ZIPs
zip_folder   = "/scratch/alice/b/bg205/smoking_data_input/Observation"
# Excel workbook with medcodeids (sheet names must match EXCEL_SHEETS below)
excel_codes  = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/modified_LRWE_Lilly_Aurum_medcodeid_clinical biomarkers.xlsx"
# TXT with Total Cholesterol medcodeids (tab-delimited or autodetected)
tot_chol_txt = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/Codelist_Total_Cholesterol.txt"
# Where to write chunk outputs
output_folder = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/test_biomarker_tmp_outputs"

# Exact sheet names (must match your workbook)
EXCEL_SHEETS = {
    "hba1c"        : "HbA1c - final",
    "hdl"          : "HDL_final",
    "ldl"          : "LDL_final",
    "triglycerides": "Trig_final",
}

# ==============================
# HELPERS
# ==============================
def die(msg, code=1):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)

def pick_medcode_column(columns):
    """
    Return the first column name that looks like a medcode column.
    Accepts things like 'medcodeid', 'medcode_id', 'code', etc.
    """
    lowered = [c for c in columns]
    # prefer explicit medcode columns
    for c in lowered:
        cl = c.lower()
        if "medcode" in cl:
            return c
    # fallback to a plain 'code' column if present
    for c in lowered:
        if c.lower() == "code":
            return c
    raise ValueError("No medcode/‘code’ column found.")

def load_medcodes_from_excel(xlsx_path, sheet_name):
    """
    Load medcodeids from an Excel tab that may have banner/title rows.
    Tries several rows as header until a column containing 'medcode' or 'code' is found.
    """
    for hdr in (0, 1, 2, 3, 4, 5):
        try:
            df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str, header=hdr)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        df.columns = [str(c).strip() for c in df.columns]
        try:
            col = pick_medcode_column(df.columns)
            return df[col].dropna().astype(str).str.strip().unique().tolist()
        except Exception:
            continue
    # absolute fallback: raw read, first column
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str, header=None)
    first_col = df.columns[0]
    return df[first_col].dropna().astype(str).str.strip().unique().tolist()

def load_totchol_from_txt(txt_path):
    """
    Load Total Cholesterol medcodeids from TXT (tab or other delimiter).
    Accepts a column named 'code' or anything containing 'medcode'.
    """
    try:
        df = pd.read_csv(txt_path, sep="\t", dtype=str)
    except Exception:
        # auto-detect delimiter
        df = pd.read_csv(txt_path, sep=None, engine="python", dtype=str)
    if df is None or df.empty:
        return []
    df.columns = [str(c).strip() for c in df.columns]
    col = pick_medcode_column(df.columns)
    return df[col].dropna().astype(str).str.strip().unique().tolist()

def find_unit_column(df):
    """
    Return the unit column name if present, else None.
    Aurum often uses 'valueunitid'/'valueunitsid', but be permissive.
    """
    candidates = (
        "valueunitid", "valueunitsid", "unit", "value_unit",
        "unitid", "valueunit", "value_unitsid"
    )
    for u in candidates:
        if u in df.columns:
            return u
    return None

# ==============================
# MAIN
# ==============================
def main():
    if len(sys.argv) < 2:
        die("Usage: python 7_test.py <zip_index>")

    try:
        zip_index = int(sys.argv[1])
    except ValueError:
        die("zip_index must be an integer (0-based)")

    # Sanity checks
    if not os.path.isdir(zip_folder):
        die(f"zip_folder not found: {zip_folder}")
    if not os.path.isfile(excel_codes):
        die(f"Excel codes file not found: {excel_codes}")
    if not os.path.isfile(tot_chol_txt):
        die(f"Total Cholesterol TXT file not found: {tot_chol_txt}")

    os.makedirs(output_folder, exist_ok=True)

    # Enumerate ZIPs
    zip_files = sorted(
        os.path.join(zip_folder, f) for f in os.listdir(zip_folder) if f.endswith(".zip")
    )
    if not zip_files:
        die(f"No .zip files found in {zip_folder}")
    if not (0 <= zip_index < len(zip_files)):
        die(f"zip_index out of range (0..{len(zip_files)-1})")

    zip_path = zip_files[zip_index]
    print(f"Processing ZIP {zip_index+1}/{len(zip_files)}: {os.path.basename(zip_path)}")

    # Load medcode sets
    codes = {}
    for var, sheet in EXCEL_SHEETS.items():
        try:
            codes[var] = load_medcodes_from_excel(excel_codes, sheet)
        except Exception as e:
            die(f"Failed reading sheet '{sheet}' for {var}: {e}")
    try:
        codes["tot_chol"] = load_totchol_from_txt(tot_chol_txt)
    except Exception as e:
        die(f"Failed reading Total Cholesterol TXT: {e}")

    for k, v in codes.items():
        print(f"  {k:13s}: {len(v):6d} codes")

    # Collect matches per variable
    tmp_records = {k: [] for k in codes}

    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if not name.lower().endswith(".txt"):
                continue
            with z.open(name) as f:
                try:
                    df = pd.read_csv(f, sep="\t", dtype=str)
                except Exception as e:
                    print(f"  Skipping {name} (read error): {e}")
                    continue

                # required columns check
                needed = {"patid", "obsdate", "medcodeid", "value"}
                if not needed.issubset(df.columns):
                    missing = ", ".join(sorted(needed - set(df.columns)))
                    print(f"  Skipping {name}: missing columns [{missing}]")
                    continue

                # normalize types
                df["medcodeid"] = df["medcodeid"].astype(str).str.strip()
                unit_col = find_unit_column(df)

                # filter and collect
                for var, medcodes in codes.items():
                    if not medcodes:
                        continue
                    matched = df[df["medcodeid"].isin(medcodes)]
                    if matched.empty:
                        continue
                    keep = ["patid", "obsdate", "medcodeid", "value"]
                    if unit_col:
                        keep.append(unit_col)
                        matched = matched[keep].rename(columns={unit_col: "unit"})
                    else:
                        matched = matched[keep]
                        matched["unit"] = pd.NA  # ensure a unit column exists downstream
                    tmp_records[var].append(matched)

    # Write outputs
    wrote_any = False
    for var, dfs in tmp_records.items():
        if not dfs:
            print(f"  No matches for {var} in this ZIP")
            continue
        out = pd.concat(dfs, ignore_index=True)
        out_file = os.path.join(output_folder, f"{var}_chunk_{zip_index:04d}.csv.gz")
        out.to_csv(out_file, index=False, compression="gzip")
        print(f"  Wrote {len(out):,} rows -> {out_file}")
        wrote_any = True

    if not wrote_any:
        print("  No biomarker rows matched in this ZIP.")

if __name__ == "__main__":
    main()
