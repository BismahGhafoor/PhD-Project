import pandas as pd
import zipfile
import os
import sys

# ------------------------
# Config
# ------------------------
zip_index = int(sys.argv[1])
zip_folder = "/scratch/alice/b/bg205/smoking_data_input/Observation"
code_folder = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/Codes/clinical_biomarkers_CSV_exports"
output_folder = "/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/biomarker_tmp_outputs_txt"

os.makedirs(output_folder, exist_ok=True)

# ------------------------
# Get list of ZIPs in order
# ------------------------
zip_files = sorted(
    os.path.join(zip_folder, f)
    for f in os.listdir(zip_folder)
    if f.endswith(".zip")
)
if not zip_files:
    raise FileNotFoundError(f"No .zip files found in {zip_folder}")
if not (0 <= zip_index < len(zip_files)):
    raise IndexError(f"zip_index {zip_index} out of range 0..{len(zip_files)-1}")

zip_path = zip_files[zip_index]

# ------------------------
# Load medcodeid lists for each biomarker
# ------------------------
def load_medcodes(file):
    path = os.path.join(code_folder, file)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # many CPRD exports have 2 header rows; try 2 then 0
    for skip in (2, 0):
        df = pd.read_csv(path, dtype=str, skiprows=skip)
        cols = df.columns.str.lower().str.strip()
        med = [c for c in cols if "medcode" in c]
        if med:
            col = df.columns[cols.get_loc(med[0])]
            return df[col].dropna().astype(str).str.strip().tolist()
    raise ValueError(f"No medcode column found in {file}")

codes = {
    "bmi":    load_medcodes("BMI_-_final.csv"),
    "weight": load_medcodes("Weight_final.csv"),
    "height": load_medcodes("Height_-_final.csv"),
    "sbp":    load_medcodes("SBP_final.csv"),
    "dbp":    load_medcodes("DBP_final.csv"),
}

# ------------------------
# Helpers
# ------------------------
def pick_value_column(df):
    # prefer 'value', fallback to common variants
    for cand in ["value", "value1", "numericvalue", "value_num", "val"]:
        if cand in df.columns:
            return cand
    return None

# ------------------------
# Process zip file
# ------------------------
print(f"Processing file {zip_index + 1}/{len(zip_files)}: {os.path.basename(zip_path)}")

tmp_records = {k: [] for k in codes}

with zipfile.ZipFile(zip_path, "r") as z:
    for name in z.namelist():
        if not name.lower().endswith(".txt"):
            continue
        with z.open(name) as f:
            try:
                df = pd.read_csv(f, sep="\t", dtype=str, low_memory=False)
            except Exception as e:
                print(f"Failed to read {name} in {zip_path}: {e}")
                continue

        if df.empty or "medcodeid" not in df.columns:
            print(f"No 'medcodeid' in {name}")
            continue
        if "patid" not in df.columns or "obsdate" not in df.columns:
            print(f"Missing patid/obsdate in {name}; skipping")
            continue

        val_col = pick_value_column(df)
        if val_col is None:
            # still keep the record skeleton with empty value
            df["value"] = pd.NA
            val_col = "value"

        df["medcodeid"] = df["medcodeid"].astype(str).str.strip()

        for var, medcodes in codes.items():
            matched = df[df["medcodeid"].isin(medcodes)]
            if not matched.empty:
                keep = matched[["patid", "obsdate", "medcodeid", val_col]].rename(columns={val_col: "value"})
                tmp_records[var].append(keep)

# ------------------------
# Save outputs (TSV .txt.gz)
# ------------------------
for var, dfs in tmp_records.items():
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        output_file = os.path.join(output_folder, f"{var}_chunk_{zip_index:04d}.txt.gz")
        result.to_csv(output_file, sep="\t", index=False, compression="gzip")
        print(f"Saved {len(result):,} rows to {output_file} ({var})")
    else:
        print(f"No {var} records found in {zip_path}")
