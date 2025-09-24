# CPRD **GOLD** — Cleaning & Feature Engineering (Scripts)

This folder contains the modular Python scripts that build an **analysis-ready GOLD cohort** and derive risk-factor features (smoking, BMI/BP, labs).  
The code is written to **stream from CPRD ZIP archives** where possible and to work on HPC or local machines.

> ⚠️ **Restricted data**: CPRD/HES/ONS files are not included. Configure paths to your secure storage before running.

---

## Quick run (typical order)

```bash
# 1) Filter diabetes codes for GOLD (keeps types 1 & 2 only)
python 1_medcode_filtering.py

# 2) Append/clean raw GOLD extracts (reads from *.zip)
python 2_Gold_data_append.py

# 3) Build baseline from appended clinical chunks + map to diabetes type
python 3_Gold_baseline.py

# 4) Add gender, YOB, ethnicity, IMD & date of death to the baseline
python 4_Baseline_dod,imdb,gen,yob,eth.py

# 5) Build clinical smoking file from clinical zips (gzipped output)
python 5_smoking_Data_append.py

# 6) From Additional Clinical files (+ HES), derive smoking, BMI and blood pressure
python 6_smoke_bmi_bp.py

# 7) Combine all Test extracts (labs) to one gz file
python 7_test_data_append.py

# 8) From the combined Test file, extract lipids & HbA1c and merge to patient
python 8_test_data_cleaning.py
```

## Environment & dependencies
Python 3.11+

`pandas`, `numpy`, `pyyaml` (if using configs), `tqdm` (optional), `openpyxl` (for Excel), `zipfile` (stdlib)

Some scripts import `helper_functions` (e.g., save_long_format_data, nperc_counts, etc.).

Ensure helper_functions.py is importable (same folder or on PYTHONPATH).

### Example 
```bash
conda create -n goldclean python=3.11 -y
conda activate goldclean
pip install pandas numpy openpyxl
```

## Inputs (where the scripts expect data)

### GOLD raw extracts (prefer zipped):

`.../GOLD/FZ_GOLD_All_Extract_Clinical_*.zip`

`.../GOLD/FZ_GOLD_All_Extract_Therapy_*.zip`

`.../GOLD/FZ_GOLD_All_Extract_Test_*.zip`

`.../GOLD/FZ_GOLD_All_Extract_Additional_*.zip`

### Linkage files (examples as used in code):

`hes_patient_23_002869_DM.txt`, `patient_2019_imd_23_002869.txt`, `death_patient_23_002869_DM.txt`

### Codelists:

`Shared/Codes/GOLD_final.txt` → filtered to `filtered_diabetes_codes.txt`

`GOLD_Codes_FZ.xlsx` (sheet Smok) for smoking medcodes

SUM.txt (HbA1c unit lookup) for `8_test_data_cleaning.py`

## Outputs (key files)

`filtered_diabetes_codes.txt` (tab-delimited)

Appended chunks:

- `Cleaned_GOLD_Extract_Clinical_*.txt`
- `Cleaned_GOLD_Extract_Therapy_*.txt`
- `Cleaned_GOLD_Extract_Test_*.txt`

Baselines:
- `baseline_ungrouped_df_WithNA.txt`
- `baseline_Type_1_Diabetes_WithNA.txt`
- `baseline_Type_2_Diabetes_WithNA.txt`
- `baseline_grouped_df_WithNA.txt`
- `baseline_with_all_features.txt` (after demographics/ethnicity/IMD/DoD merge)

Smoking:
- `Clinical_SmokingStatus_all.txt.gz`
- `Cleaned_Patient_Smoking_Data.csv` (after adding BMI/BP as well)

Tests/Labs:
- `Test_entities_all.txt.gz`
- `extracted_lab_data.csv` (final merged labs incl. HbA1c/lipids)

Folder names such as `backup_chunks/` may be created by scripts.

## Script-by-script details
### 1) 1_medcode_filtering.py
- Reads `GOLD_final.txt` (tab-delimited with medcode, type).
- Keeps type 1 and 2 only; renames medcode→code and adds terminology='medcode'.
- Output: `filtered_diabetes_codes.txt`.

### 2) 2_Gold_data_append.py
- Streams CPRD Clinical/Test/Therapy zips.
- Optionally filters by codelists (clinical: medcode/enttype; therapy: prodcode).
- Writes chunked TSVs:
-   `Cleaned_GOLD_Extract_Clinical_{chunk}.txt`
-   `Cleaned_GOLD_Extract_Therapy_{chunk}.txt`
-   `Cleaned_GOLD_Extract_Test_{chunk}.txt`
Tunables: filter_* flags and max_rows_limit (chunk size).

### 3) 3_Gold_baseline.py
- Backs up clinical chunks to backup_chunks/ then builds a baseline:
- Keeps patid, earliest eventdate, and medcode.
- Maps medcode → diabetes_type via `filtered_diabetes_codes.txt`.
- Saves:
-   `baseline_ungrouped_df_WithNA.txt`
-   grouped files per type (Type_1, Type_2) and baseline_grouped_df_WithNA.txt.
-   Verifies earliest-date logic across chunks.

### 4) 4_Baseline_dod,imdb,gen,yob,eth.py
- Merges gender/YOB (Patient), ethnicity (HES), IMD, and date of death onto the baseline.
- Supports zipped patient extract via a helper that reads the first .txt inside the .zip.
- Output: `baseline_with_all_features.txt`.

### 5) 5_smoking_Data_append.py
- From Clinical zips, filters to smoking medcodes listed in GOLD_Codes_FZ.xlsx (sheet Smok).
- Keeps minimal columns and gzips the result.
- Output: `Clinical_SmokingStatus_all.txt.gz`.

### 6) 6_smoke_bmi_bp.py
- Streams Additional Clinical files (prefers zips) in chunks, writing temporary CSVs per subset:
-   BP (enttype == "1"), smoking ("4"), weight ("13"), height ("14").
-   Loads HES diagnoses to add ICD-based smoking signal.
-   Functions:
-   `get_smoking_data(...)` → combines clinical medcode, additional enttype and HES into smoking_status.
-   `weight_height_bmi(...)` → derives BMI from weight/height (cleans and bounds).
-   `get_bp_data(...)` → derives systolic/diastolic with plausibility bounds.
-  Output: `Cleaned_Patient_Smoking_Data.csv` (patient + smoking + BMI + BP).

### 7) 7_test_data_append.py
- Streams Test zips in chunks, parses dates, computes per-chunk indexdate (min eventdate per patid),
keeps patid,eventdate,indexdate,enttype,data1→value,data2→unit.
Output: `Test_entities_all.txt.gz`.

### 8) 8_test_data_cleaning.py
- Reads Cleaned_Patient_Smoking_Data.csv and Test_entities_all.txt.gz (filtered to cohort patids).
- Extracts labs without date-window filtering; selects the record closest to indexdate:
- Total cholesterol (enttype="163"), HDL ("175"), LDL ("177"), Triglycerides ("202"), HbA1c ("275", with unit-aware conversion to % using SUM.txt).
- Output: `extracted_lab_data.csv` (in working directory).

## Configuration & paths
Paths are defined at the top of the scripts (e.g., *_files_directory, linkage file locations).

Edit these first to match your secure directories. For portability, consider moving paths to a config_gold.yaml and reading them with pyyaml.

## Performance notes

All “append” scripts use chunking (default: max_rows_limit=20000). Increase/decrease per your RAM.

Zipped reading avoids expanding multi-GB files on disk.

For HPC: wrap calls in .sbatch and set environment variables for paths/chunks if needed.

## Troubleshooting

No files found: Confirm the `*_pattern` globs and that you have read access to the secure mount.

Mixed extensions error: Ensure a folder contains either `.zip` or `.txt`, not both, or allow mixed handling where scripted.

Missing columns: Different CPRD exports can vary—print `df.columns` and adjust `usecols` / column names.

Date parsing: All scripts assume day-first `(%d/%m/%Y)`. If yours differ, update the `to_datetime` calls.

helper_functions import: Put `helper_functions.py` next to these scripts or add its folder to `PYTHONPATH`.



