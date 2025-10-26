#!/usr/bin/env python3
# Extract HDL, LDL, Triglycerides, HbA1c (from Excel tabs) and Total Cholesterol (from TXT)
# from CPRD Aurum Observation ZIPs into per-ZIP gzipped TSV chunks.

import os
import sys
import zipfile
import pandas as pd

# ==============================
# CONFIG — EDIT THESE PATHS ONLY
# ==============================
zip_folder    = "/scratch/alice/b/bg205/smoking_data_input/Observation"
excel_codes   = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/modified_LRWE_Lilly_Aurum_medcodeid_clinical biomarkers.xlsx"
tot_chol_txt  = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/Codelist_Total_Cholesterol.txt"
output_folder = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/tests_tmp_outputs_txt"  # <- TSV chunks here

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
    print(f"ERROR: {msg}", file=sys.stderr); sys.exit(code)

def pick_medcode_column(columns):
    for c in columns:
        if "medcode" in str(c).lower():
            return c
    for c in columns:
        if str(c).lower() == "code":
            return c
    raise ValueError("No medcode/‘code’ column found.")

def load_medcodes_from_excel(xlsx_path, sheet_name):
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
            return (df[col].dropna().astype(str).str.strip().unique().tolist())
        except Exception:
            continue
    # fallback: first column
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str, header=None)
    first_col = df.columns[0]
    return df[first_col].dropna().astype(str).str.strip().unique().tolist()

def load_totchol_from_txt(txt_path):
    try:
        df = pd.read_csv(txt_path, sep="\t", dtype=str)
    except Exception:
        df = pd.read_csv(txt_path, sep=None, engine="python", dtype=str)
    if df is None or df.empty:
        return []
    df.columns = [str(c).strip() for c in df.columns]
    col = pick_medcode_column(df.columns)
    return df[col].dropna().astype(str).str.strip().unique().tolist()

def find_unit_column(df):
    candidates = (
        "valueunitid", "valueunitsid", "unit", "value_unit",
        "unitid", "valueunit", "value_unitsid"
    )
    for u in candidates:
        if u in df.columns:
            return u
    return None

def pick_value_column(df):
    """Return the first present numeric value column name."""
    for c in ("value", "value1", "value2", "numvalue"):
        if c in df.columns:
            return c
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

    # checks
    if not os.path.isdir(zip_folder):      die(f"zip_folder not found: {zip_folder}")
    if not os.path.isfile(excel_codes):    die(f"Excel codes file not found: {excel_codes}")
    if not os.path.isfile(tot_chol_txt):   die(f"Total Cholesterol TXT file not found: {tot_chol_txt}")
    os.makedirs(output_folder, exist_ok=True)

    # list zips
    zip_files = sorted(os.path.join(zip_folder, f) for f in os.listdir(zip_folder) if f.endswith(".zip"))
    if not zip_files: die(f"No .zip files found in {zip_folder}")
    if not (0 <= zip_index < len(zip_files)): die(f"zip_index out of range (0..{len(zip_files)-1})")

    zip_path = zip_files[zip_index]
    print(f"Processing ZIP {zip_index+1}/{len(zip_files)}: {os.path.basename(zip_path)}")

    # load medcodes
    codes = {}
    for var, sheet in EXCEL_SHEETS.items():
        codes[var] = load_medcodes_from_excel(excel_codes, sheet)
    codes["tot_chol"] = load_totchol_from_txt(tot_chol_txt)
    for k, v in codes.items():
        print(f"  {k:13s}: {len(v):6d} codes")

    tmp_records = {k: [] for k in codes}

    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if not name.lower().endswith(".txt"):
                continue
            with z.open(name) as f:
                try:
                    # read only potentially-needed columns
                    df = pd.read_csv(
                        f, sep="\t", dtype=str, low_memory=False,
                        usecols=lambda c: str(c).lower() in {
                            "patid","obsdate","medcodeid",
                            "value","value1","value2","numvalue",
                            "valueunitid","valueunitsid","unit","value_unit","unitid","valueunit","value_unitsid"
                        }
                    )
                except Exception as e:
                    print(f"  Skipping {name} (read error): {e}")
                    continue

                need_base = {"patid","obsdate","medcodeid"}
                if not need_base.issubset(df.columns):
                    missing = ", ".join(sorted(need_base - set(df.columns)))
                    print(f"  Skipping {name}: missing columns [{missing}]")
                    continue

                df["medcodeid"] = df["medcodeid"].astype(str).str.strip()

                val_col  = pick_value_column(df)
                unit_col = find_unit_column(df)

                if val_col is None:
                    print(f"  Skipping {name}: no numeric value column (value/value1/value2/numvalue)")
                    continue

                base_keep = ["patid","obsdate","medcodeid", val_col]
                if unit_col: base_keep.append(unit_col)

                for var, medcodes in codes.items():
                    if not medcodes: 
                        continue
                    matched = df[df["medcodeid"].isin(medcodes)]
                    if matched.empty:
                        continue
                    out = matched[base_keep].copy()
                    out = out.rename(columns={val_col: "value"})
                    if unit_col:
                        out = out.rename(columns={unit_col: "unit"})
                    else:
                        out["unit"] = pd.NA
                    tmp_records[var].append(out)

    wrote_any = False
    for var, dfs in tmp_records.items():
        if not dfs:
            print(f"  No matches for {var} in this ZIP")
            continue
        out = pd.concat(dfs, ignore_index=True)
        out_file = os.path.join(output_folder, f"{var}_chunk_{zip_index:04d}.txt.gz")
        out.to_csv(out_file, sep="\t", index=False, compression="gzip")
        print(f"  Wrote {len(out):,} rows -> {out_file}")
        wrote_any = True

    if not wrote_any:
        print("  No biomarker rows matched in this ZIP.")

if __name__ == "__main__":
    main()
