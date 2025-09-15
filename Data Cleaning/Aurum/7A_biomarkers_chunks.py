import pandas as pd
import zipfile
import os
import sys

# ------------------------
# Config
# ------------------------
zip_index = int(sys.argv[1])
zip_folder = "/scratch/alice/b/bg205/smoking_data_input/Observation"
code_folder = "/scratch/alice/b/bg205/smoking_run/clinical_biomarkers_CSV_exports"
output_folder = "biomarker_tmp_outputs"

os.makedirs(output_folder, exist_ok=True)

# ------------------------
# Get list of ZIPs in order
# ------------------------
zip_files = sorted([
    os.path.join(zip_folder, f)
    for f in os.listdir(zip_folder)
    if f.endswith(".zip")
])

zip_path = zip_files[zip_index]

# ------------------------
# Load medcodeid lists for each biomarker
# ------------------------
def load_medcodes(file):
    df = pd.read_csv(os.path.join(code_folder, file), skiprows=2)
    df.columns = df.columns.str.lower().str.strip()
    medcode_col = [col for col in df.columns if 'medcode' in col][0]
    return df[medcode_col].dropna().astype(str).str.strip().tolist()

codes = {
    'bmi': load_medcodes("BMI_-_final.csv"),
    'weight': load_medcodes("Weight_final.csv"),
    'height': load_medcodes("Height_-_final.csv"),
    'sbp': load_medcodes("SBP_final.csv"),
    'dbp': load_medcodes("DBP_final.csv"),
}

# ------------------------
# Process zip file
# ------------------------
print(f"Processing file {zip_index + 1}/{len(zip_files)}: {os.path.basename(zip_path)}")

tmp_records = {k: [] for k in codes}

with zipfile.ZipFile(zip_path, 'r') as z:
    for name in z.namelist():
        if name.endswith(".txt"):
            with z.open(name) as f:
                try:
                    df = pd.read_csv(f, sep="\t", dtype=str)
                except Exception as e:
                    print(f"Failed to read {name} in {zip_path}: {e}")
                    continue

                if 'medcodeid' not in df.columns:
                    print(f"No 'medcodeid' column in {name}")
                    continue

                df['medcodeid'] = df['medcodeid'].astype(str).str.strip()

                for var, medcodes in codes.items():
                    matched = df[df['medcodeid'].isin(medcodes)]
                    if not matched.empty:
                        matched = matched[['patid', 'obsdate', 'medcodeid', 'value']]
                        tmp_records[var].append(matched)

# ------------------------
# Save outputs
# ------------------------
for var, dfs in tmp_records.items():
    if dfs:
        result = pd.concat(dfs, ignore_index=True)
        output_file = os.path.join(output_folder, f"{var}_chunk_{zip_index:04d}.csv.gz")
        result.to_csv(output_file, index=False, compression="gzip")
        print(f"Saved {len(result)} rows to {output_file} ({var})")
    else:
        print(f"No {var} records found in {zip_path}")
