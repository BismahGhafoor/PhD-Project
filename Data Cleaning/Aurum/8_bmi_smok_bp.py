import pandas as pd
import numpy as np
import glob
import os

print("Setting up file paths...")

# --------------------------
# Paths
# --------------------------
biomarker_chunk_folder = "biomarker_tmp_outputs"
clinical_smoking_file = "Aurum_smoking_records_all.csv.gz"
demographics_file = "Enriched_baseline_with_demographics.csv"
hes_file = "hes_diagnosis_hosp_23_002869_DM.txt"
medcode_csv_path = "/scratch/alice/b/bg205/smoking_run/clinical_biomarkers_CSV_exports"

# --------------------------
# Load demographic and HES data
# --------------------------
print("Loading demographics and clinical smoking data...")
#demog = pd.read_csv(demographics_file, dtype=str, parse_dates=['indexdate', 'dod'], dayfirst=True)
demog = pd.read_csv(demographics_file, dtype=str, dayfirst=True)
demog['indexdate'] = pd.to_datetime(demog['indexdate'], errors='coerce')
demog['dod'] = pd.to_datetime(demog['dod'], errors='coerce')
demog['patid'] = demog['patid'].astype(str)

print("[DEBUG] Sample loaded demographics:")
print(demog[['patid', 'indexdate']].head(10))
print(f"[DEBUG] Missing indexdate: {demog['indexdate'].isna().sum()}")

clinical_smok = pd.read_csv(clinical_smoking_file, sep=",", compression="gzip", parse_dates=['obsdate'], dayfirst=True)

print("Loading and merging HES data in chunks...")
chunked_hes = []
reader = pd.read_csv(hes_file, sep="\t", dtype=str, chunksize=500_000)
for i, chunk in enumerate(reader):
    print(f"  Processing HES chunk {i+1}")
    chunk['admidate'] = pd.to_datetime(chunk['admidate'], errors='coerce', dayfirst=True)
    chunk['patid'] = chunk['patid'].astype(str)
    merged = chunk.merge(demog[['patid', 'indexdate']], on='patid', how='left')
    chunked_hes.append(merged)
hes_hosp = pd.concat(chunked_hes, ignore_index=True)

# --------------------------
# Load pre-extracted biomarker chunks
# --------------------------
def load_all_chunks(prefix):
    print(f"Loading {prefix} chunks...")
    files = sorted(glob.glob(os.path.join(biomarker_chunk_folder, f"{prefix}_chunk_*.csv.gz")))
    return pd.concat((pd.read_csv(f, dtype=str) for f in files), ignore_index=True)

obs_data = {
    'bmi': load_all_chunks("bmi"),
    'weight': load_all_chunks("weight"),
    'height': load_all_chunks("height"),
    'bp': pd.concat([
        load_all_chunks("sbp"),
        load_all_chunks("dbp")
    ], ignore_index=True)
}

# --------------------------
# Smoking variable
# --------------------------
print("Extracting and merging smoking status...")

def load_codes(file):
    path = os.path.join(medcode_csv_path, file)
    
    # If the filename ends in ".cleaned.csv", don't skip rows
    skip = 2 if not file.endswith(".cleaned.csv") else 0
    
    df = pd.read_csv(path, skiprows=skip)
    df.columns = df.columns.str.lower().str.strip()
    
    medcode_cols = [c for c in df.columns if 'medcode' in c]
    if not medcode_cols:
        raise ValueError(f"No column with 'medcode' in {file}. Columns: {df.columns.tolist()}")
    
    return df[medcode_cols[0]].dropna().astype(str).str.strip().tolist()

def get_smoking_data(clinical_smok, hes_hosp, patient):
    """
    Build smoking status with CPRD priority, HES fallback.
    - CPRD: maps medcodeid lists for current/ex/never and picks record closest to indexdate.
    - HES: maps ICD codes {F17*, Z72.0, T65.2} -> 'Yes' (current smoker), picks closest to indexdate.
    - Merge: keep CPRD when present; only use HES for patients with no CPRD smoking record.
    """
    # ---------------- CPRD ----------------
    current_codes = load_codes("Current_smoker.csv")
    ex_codes      = load_codes("Ex-smoker.csv")
    never_codes   = load_codes("Never_smoked.csv")

    smoking_clin = clinical_smok.copy()
    smoking_clin['patid']     = smoking_clin['patid'].astype(str)
    smoking_clin['medcodeid'] = smoking_clin['medcodeid'].astype(str)
    smoking_clin = smoking_clin.rename(columns={'obsdate': 'smoking_date'})
    smoking_clin['smoking_status'] = pd.Series(dtype='str')

    smoking_clin.loc[smoking_clin['medcodeid'].isin(current_codes), 'smoking_status'] = 'Yes'
    smoking_clin.loc[smoking_clin['medcodeid'].isin(ex_codes),      'smoking_status'] = 'Ex'
    smoking_clin.loc[smoking_clin['medcodeid'].isin(never_codes),   'smoking_status'] = 'No'

    # attach indexdate and pick nearest per (patid,indexdate)
    smoking_clin = smoking_clin.merge(patient[['patid','indexdate']], on='patid', how='left')
    smoking_clin = smoking_clin[['patid','indexdate','smoking_date','smoking_status']].dropna(subset=['smoking_status'])
    smoking_clin['indexdate']    = pd.to_datetime(smoking_clin['indexdate'], errors='coerce')
    smoking_clin['smoking_date'] = pd.to_datetime(smoking_clin['smoking_date'], errors='coerce')
    smoking_clin['gap'] = (smoking_clin['indexdate'] - smoking_clin['smoking_date']).abs().dt.days
    smoking_clin = smoking_clin.sort_values('gap').drop_duplicates(['patid','indexdate'])
    smoking_clin['smoking_source'] = 'cprd'

    # ---------------- HES ----------------
    # Map ICD codes to 'Yes' (current smoker). HES doesn’t give reliable Ex/Never flags here.
    hes = hes_hosp.copy()
    hes['patid'] = hes['patid'].astype(str)
    icd_col = 'ICD' if 'ICD' in hes.columns else ('diag_icd10' if 'diag_icd10' in hes.columns else None)
    if icd_col is None:
        raise KeyError("No ICD column found in HES file (expected 'ICD' or 'diag_icd10').")
    hes['ICD'] = hes[icd_col].astype(str).str.strip().str.upper()
    hes['admidate'] = pd.to_datetime(hes['admidate'], errors='coerce', dayfirst=True)


    # ICD mapping: F17* (nicotine dependence), Z72.0 (tobacco use), T65.2 (toxic effect of tobacco)
    def hes_is_current(icd):
        if pd.isna(icd): 
            return False
        icd = str(icd).upper()
        return icd.startswith('F17') or icd == 'Z72.0' or icd == 'T65.2'

    hes['is_current'] = hes['ICD'].apply(hes_is_current)
    hes_smoking = hes[hes['is_current']].copy()
    hes_smoking['smoking_status'] = 'Yes'
    hes_smoking = hes_smoking[['patid','admidate','smoking_status']].rename(columns={'admidate':'smoking_date'})

    # attach indexdate and pick nearest per (patid,indexdate)
    if not hes_smoking.empty:
        hes_smoking = hes_smoking.merge(patient[['patid','indexdate']], on='patid', how='left')
        hes_smoking['indexdate']    = pd.to_datetime(hes_smoking['indexdate'], errors='coerce')
        hes_smoking['smoking_date'] = pd.to_datetime(hes_smoking['smoking_date'], errors='coerce')
        hes_smoking = hes_smoking.dropna(subset=['indexdate','smoking_date'])
        hes_smoking['gap'] = (hes_smoking['indexdate'] - hes_smoking['smoking_date']).abs().dt.days
        hes_smoking = hes_smoking.sort_values('gap').drop_duplicates(['patid','indexdate'])
        hes_smoking['smoking_source'] = 'hes'
    else:
        hes_smoking = pd.DataFrame(columns=['patid','indexdate','smoking_date','smoking_status','smoking_source'])

    # ---------------- Merge: CPRD priority, HES fallback ----------------
    # Start with CPRD
    combined = smoking_clin.copy()

    # Add HES rows only where no CPRD record exists for (patid,indexdate)
    if not hes_smoking.empty:
        keys = ['patid','indexdate']
        hes_only = hes_smoking.merge(combined[keys], on=keys, how='left', indicator=True)
        hes_only = hes_only[hes_only['_merge'] == 'left_only'].drop(columns=['_merge'])
        combined = pd.concat([combined, hes_only], ignore_index=True, sort=False)

    # Final merge back to patient table
    patient = patient.copy()
    patient['indexdate'] = pd.to_datetime(patient['indexdate'], errors='coerce')
    combined['indexdate'] = pd.to_datetime(combined['indexdate'], errors='coerce')

    return patient.merge(
        combined[['patid','indexdate','smoking_date','smoking_status','smoking_source']],
        on=['patid','indexdate'],
        how='left'
    )

demog = get_smoking_data(clinical_smok, hes_hosp, demog)
print("Smoking status merged.")

# --------------------------
# BMI calculation
# --------------------------
print("Processing BMI, weight, and height...")

# Copy original data
bmi_obs = obs_data['bmi'].copy()
weight = obs_data['weight'].copy()
height = obs_data['height'].copy()

# Format and merge: BMI
print("Cleaning recorded BMI observations...")
bmi_obs['obsdate'] = pd.to_datetime(bmi_obs['obsdate'], errors='coerce', dayfirst=True)
bmi_obs['medcodeid'] = bmi_obs['medcodeid'].astype(str)
bmi_obs = bmi_obs.merge(demog[['patid', 'indexdate']], on='patid', how='left')
bmi_obs['value'] = pd.to_numeric(bmi_obs['value'], errors='coerce')

# Format and merge: weight
print("Cleaning weight observations...")
weight['obsdate'] = pd.to_datetime(weight['obsdate'], errors='coerce', dayfirst=True)
weight['medcodeid'] = weight['medcodeid'].astype(str)
weight = weight.merge(demog[['patid', 'indexdate']], on='patid', how='left')
weight['value'] = pd.to_numeric(weight['value'], errors='coerce')

# Format and merge: height
print("Cleaning height observations...")
height['obsdate'] = pd.to_datetime(height['obsdate'], errors='coerce', dayfirst=True)
height['medcodeid'] = height['medcodeid'].astype(str)
height = height.merge(demog[['patid', 'indexdate']], on='patid', how='left')
height['value'] = pd.to_numeric(height['value'], errors='coerce')

# Extract valid recorded BMI
print("Extracting valid recorded BMI values...")
recorded_bmi = bmi_obs[['patid', 'obsdate', 'indexdate', 'value']].rename(columns={'value': 'bmi'})
recorded_bmi = recorded_bmi[(recorded_bmi['bmi'] >= 10) & (recorded_bmi['bmi'] <= 70)]
recorded_bmi['gap'] = (recorded_bmi['indexdate'] - recorded_bmi['obsdate']).abs().dt.days
recorded_bmi = recorded_bmi.sort_values('gap').drop_duplicates(['patid', 'indexdate'])

print(f"Recorded BMI values found for {recorded_bmi['patid'].nunique()} patients.")

# Calculate BMI from weight and height
print("Calculating BMI from weight and height...")
weight = weight.rename(columns={'value': 'weight_kg'})[['patid', 'obsdate', 'indexdate', 'weight_kg']]
height = height.rename(columns={'value': 'height_m'})[['patid', 'obsdate', 'indexdate', 'height_m']]
bmi_calc = pd.merge(weight, height, on=['patid', 'obsdate', 'indexdate'], how='outer')
bmi_calc['bmi'] = bmi_calc['weight_kg'] / (bmi_calc['height_m'] ** 2)
bmi_calc = bmi_calc[(bmi_calc['bmi'] >= 10) & (bmi_calc['bmi'] <= 70)]
bmi_calc['gap'] = (bmi_calc['indexdate'] - bmi_calc['obsdate']).abs().dt.days
bmi_calc = bmi_calc.sort_values('gap').drop_duplicates(['patid', 'indexdate'])

print(f"Calculated BMI values found for {bmi_calc['patid'].nunique()} patients before fallback filtering.")

# Keep calculated BMI only for patients without recorded BMI
print("Applying fallback logic: removing calculated BMI where recorded BMI exists...")
recorded_ids = set(recorded_bmi['patid'].astype(str) + "_" + recorded_bmi['indexdate'].astype(str))
bmi_calc['key'] = bmi_calc['patid'].astype(str) + "_" + bmi_calc['indexdate'].astype(str)
bmi_calc = bmi_calc[~bmi_calc['key'].isin(recorded_ids)].drop(columns='key')

print(f"Calculated BMI used as fallback for {bmi_calc['patid'].nunique()} patients.")

# Final BMI: combine and deduplicate
bmi_final = pd.concat([recorded_bmi, bmi_calc], ignore_index=True)
bmi_final = bmi_final.sort_values('gap').drop_duplicates(['patid', 'indexdate'])

# Merge into demog
demog = demog.merge(
    bmi_final[['patid', 'indexdate', 'obsdate', 'bmi']].rename(columns={'obsdate': 'bmi_date'}),
    on=['patid', 'indexdate'], how='left'
)
print("BMI data merged into demographic dataset.")

# --------------------------
# Blood Pressure
# --------------------------
print("Processing systolic and diastolic blood pressure...")

# Copy and clean observation data
bp = obs_data['bp'].copy()
bp['obsdate'] = pd.to_datetime(bp['obsdate'], errors='coerce', dayfirst=True)
bp['medcodeid'] = bp['medcodeid'].astype(str)
bp = bp.merge(demog[['patid', 'indexdate']], on='patid', how='left')
bp['value'] = pd.to_numeric(bp['value'], errors='coerce')

# Ensure required SBP/DBP code files exist
required_bp_files = [
    os.path.join(medcode_csv_path, "SBP_final.csv"),
    os.path.join(medcode_csv_path, "DBP_final.csv")
]
for f in required_bp_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing blood pressure medcode file: {f}")

# Clean SBP and DBP code files
print("Cleaning SBP and DBP code files...")
pd.read_csv(required_bp_files[0], skiprows=2).to_csv(
    os.path.join(medcode_csv_path, "SBP_final.cleaned.csv"), index=False)
pd.read_csv(required_bp_files[1], skiprows=2).to_csv(
    os.path.join(medcode_csv_path, "DBP_final.cleaned.csv"), index=False)

# Load cleaned code lists
print("Loading SBP and DBP medcode lists...")
sbp_codes = load_codes("SBP_final.cleaned.csv")
dbp_codes = load_codes("DBP_final.cleaned.csv")

if not sbp_codes or not dbp_codes:
    raise ValueError("SBP or DBP code list is empty after cleaning. Check CSV content.")

# Extract SBP and DBP values
print("Extracting systolic and diastolic readings from observations...")
sbp = bp[bp['medcodeid'].isin(sbp_codes)].rename(columns={'value': 'systolic'})
dbp = bp[bp['medcodeid'].isin(dbp_codes)].rename(columns={'value': 'diastolic'})

print(f"Systolic values found for {sbp['patid'].nunique()} patients.")
print(f"Diastolic values found for {dbp['patid'].nunique()} patients.")

# Merge systolic and diastolic readings on same obsdate/indexdate
print("Merging systolic and diastolic readings...")
bp_combined = pd.merge(sbp, dbp, on=['patid', 'obsdate', 'indexdate'], how='outer')

# Filter for valid keys before deduplication and merging
bp_combined = bp_combined.dropna(subset=['patid', 'indexdate', 'obsdate'])
bp_combined['patid'] = bp_combined['patid'].astype(str)
bp_combined['indexdate'] = pd.to_datetime(bp_combined['indexdate'], errors='coerce')
bp_combined['obsdate'] = pd.to_datetime(bp_combined['obsdate'], errors='coerce')

# Calculate gap and deduplicate by closest to indexdate
bp_combined['gap'] = (bp_combined['indexdate'] - bp_combined['obsdate']).abs().dt.days
bp_combined = bp_combined.sort_values('gap').drop_duplicates(['patid', 'indexdate'])

# Merge final blood pressure values into demog
demog = demog.merge(
    bp_combined[['patid', 'indexdate', 'obsdate', 'systolic', 'diastolic']].rename(columns={'obsdate': 'bp_date'}),
    on=['patid', 'indexdate'], how='left'
)

print("Blood pressure data merged into demographic dataset.")


# --------------------------
# Export
# --------------------------
print("Exporting final dataset...")

final_cols = [
    'patid', 'indexdate', 'diabetes_type', 'gender', 'yob', 'gen_ethnicity', 'e2019_imd_10', 'dod',
    'smoking_date', 'smoking_status',
    'bmi_date', 'bmi',
    'bp_date', 'systolic', 'diastolic'
]

missing_cols = [col for col in final_cols if col not in demog.columns]
if missing_cols:
    print(f"Warning: Missing columns will be filled with NaN: {missing_cols}")
    for col in missing_cols:
        demog[col] = np.nan

demog = demog[final_cols]
demog.to_csv("Enriched_Aurum_with_Biomarkers.csv", index=False)

print("Final enriched Aurum dataset saved as 'Enriched_Aurum_with_Biomarkers.csv'")
