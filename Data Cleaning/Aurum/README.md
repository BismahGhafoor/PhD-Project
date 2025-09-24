# CPRD **AURUM** — Cleaning & Feature Engineering (Scripts)

This folder contains modular Python scripts to build an **analysis-ready AURUM cohort** and derive risk-factor features (smoking, BMI/BP, labs).  
The code is written to **stream from CPRD ZIP archives** where possible and to run on HPC or local machines. SLURM array helpers are included for large-scale chunking.

> ⚠️ **Restricted data**: CPRD/HES/ONS files are not included. Point the scripts at your secure mounts before running.

---

## Quick run (typical order)

```bash
# 1) Filter diabetes codes for AURUM (keeps types 1 & 2)
python 1_medcode_filtering.py

# 2) Append/clean raw AURUM extracts (Observation & DrugIssue from *.zip)
python 2_data_append_aurum.py

# 3) Build baseline from Observation chunks + map to diabetes type
python 3_aurum_baseline_df.py

# 4) Add gender, YOB, ethnicity, IMD & date of death to the baseline
python 4_dod,imdb,gen,yob,patient.py

# 5) Extract clinical smoking from Observation ZIPs (array/parallel friendly)
#    (a) Per-ZIP extractor
python 5A_smoking_Chunks.py <zip_index|omit-for-all>
#    (b) SLURM helpers (optional)
sbatch 5B_SLURM_smoking.sbatch          # single array batch (edit template first)
bash   5C_wrapper_smoking.sh            # submits multiple batches then a merge job

# 6) Concatenate smoking chunk outputs
python 6_smoking_concatenator.py

# 7) Extract BMI/height/weight and BP codes from Observation ZIPs (array)
#    (a) Per-ZIP extractor
python 7A_biomarkers_chunks.py <zip_index>
#    (b) SLURM helpers (optional)
sbatch 7B_SLURM_biomarkers.sbatch
bash   7C_wrapper_biomarkers.sh

# 8) Merge smoking + BMI + BP with enriched baseline
python 8A_bmi_smok_bp.py
#    (SLURM wrapper)
sbatch 8B_SLURM_bmi_smok_bp.sbatch

# 9) Extract tests (HDL/LDL/TG/HbA1c/Total Chol) from Observation ZIPs (array)
#    (a) Per-ZIP extractor
python 9A_tests_chunks.py <zip_index>
#    (b) SLURM helpers (optional)
sbatch 9B_SLURM_tests_chunks.sbatch
bash   9C_wrapper_tests_chunks.sh

# 10) Clean & standardise test results and merge to cohort
python 10A_tests_datacleaning.py
#     (optional SLURM wrapper)
sbatch 10B_SLURM_tests_datacleaning.sbatch
```

## Environment & dependencies

Python 3.11+

`pandas`, `numpy`, `openpyxl` (read Excel codelists), `pyyaml` (if you externalise paths), `tqdm` (optional), `zipfile` (stdlib).
`OpenMP/BLAS` and `pyarrow` are optional but useful for IO.

Some scripts read spreadsheets exported from CPRD code browsers; headers may start on row 3—code handles this.

### Example environment creation
```bash
conda env create -f environment.yml
conda activate goldclean   # or your chosen env name
```

## Inputs (where the scripts expect data)
### AURUM raw extracts (prefer zipped)
- `.../Aurum/Observation/FZ_Aurum_1_Extract_Observation_*.zip`
- `.../Aurum/DrugIssue/*.zip`
-  `.../Aurum/Patient/*.zip` (used for gender, YOB and EMIS death date)

### Linkage files (AURUM_linked examples)
- `hes_patient_23_002869_DM.txt` (ethnicity)
- `patient_2019_imd_23_002869.txt` (IMD decile)
- `death_patient_23_002869_DM.txt` (HES death date)

### Codelists
- `Shared/Codes/aurum_final.txt` → filtered to `filtered_diabetes_AURUM_codes.txt`
- Smoking medcode CSVs (exported): Current_smoker.csv, Ex-smoker.csv, Never_smoked.csv
- Biomarker medcode CSVs: BMI_-_final.csv, Weight_final.csv, Height_-_final.csv, SBP_final.csv, DBP_final.csv

### Test medcodes:
- Excel workbook: `modified_LRWE_Lilly_Aurum_medcodeid_clinical biomarkers.xlsx` (tabs: HbA1c - final, HDL_final, LDL_final, Trig_final)
- TXT: `Codelist_Total_Cholesterol.txt` (contains a code or medcodeid column)

## Outputs (key files)
### From 1–3 (base cohort)
- `filtered_diabetes_AURUM_codes.txt`
- `filtered_aurum_chunks/`
- `Cleaned_AURUM_Observation_*.txt`
- `Cleaned_AURUM_DrugIssue_*.txt`

### Baselines:
- `aurum_baseline_ungrouped_df_NoNA.txt`
- `aurum_baseline_grouped_df_NoNA.txt`
- `aurum_baseline_Type_1_Diabetes_NoNA.txt`
- `aurum_baseline_Type_2_Diabetes_NoNA.txt`

### From 4 (demographics/eth/IMD/DoD)
- `Enriched_baseline_with_demographics.csv`

### From 5–6 (smoking)
- `Aurum_Clinical_SmokingStatus_task####.csv.gz` (per-ZIP)
- `Aurum_Clinical_SmokingStatus_all.csv.gz` (if run without array)
- `Aurum_smoking_records_all.csv.gz` (concatenated)

### From 7–8 (BMI, BP + merge)
- `biomarker_tmp_outputs/`
- `bmi_chunk_####.csv.gz`, `weight_chunk_####.csv.gz`, `height_chunk_####.csv.gz`, `sbp_chunk_####.csv.gz`, `dbp_chunk_####.csv.gz`

## From 9–10 (labs)
- `test_biomarker_tmp_outputs/`
- `hba1c_chunk_####.csv.gz`, `hdl_chunk_####.csv.gz`, `ldl_chunk_####.csv.gz`, `triglycerides_chunk_####.csv.gz`, `tot_chol_chunk_####.csv.gz`
- `FINAL_Aurum_with_Tests.csv` (cohort + standardised lipids + HbA1c)

Folder names such as `aurum_backup_chunks/`, `logs/` may be created by scripts/wrappers.

## Script-by-script details
### 1) 1_medcode_filtering.py
- Reads `aurum_final.txt` (tab-delimited with medcode, type).
- Keeps types 1 & 2; renames medcode→code, adds terminology='medcode'.
- Output: `filtered_diabetes_AURUM_codes.txt`.

### 2) 2_data_append_aurum.py
- Streams Observation and DrugIssue ZIPs; normalises TXT members.
- Optional filtering by clinical medcodes / therapy productcodeids.
- Output dir: `filtered_aurum_chunks/`
  `Cleaned_AURUM_Observation_{i}.txt`, `Cleaned_AURUM_DrugIssue_{i}.txt`.

### 3) 3_aurum_baseline_df.py
- Reads Observation chunks, keeps patid, earliest obsdate, medcodeid.
- Maps medcodeid → diabetes_type using filtered codelist.
- Saves ungrouped, grouped and type-specific baseline files.

### 4) 4_dod,imdb,gen,yob,patient.py
- Patient ZIPs → gender (decoded via lookup ZIP) and yob.
- Ethnicity: HES primary; CPRD fallback by matching ethnicity medcodes in Observation chunks.
- IMD decile and Date of Death (HES + EMIS fallback).
- Output: `Enriched_baseline_with_demographics.csv`.

### 5) 5A_smoking_Chunks.py (+ 5B_…sbatch, 5C_wrapper_smoking.sh)
- Per-ZIP extraction of smoking medcodeids (current/ex/never).
- Writes gzipped CSV chunks; wrappers submit array jobs then merge.

### 6) 6_smoking_concatenator.py
- Concatenates per-ZIP smoking chunks → `Aurum_smoking_records_all.csv.gz`.

### 7) 7A_biomarkers_chunks.py (+ 7B/7C SLURM)
- Per-ZIP extraction for BMI/weight/height/SBP/DBP medcodeids.
- Writes one gz chunk per variable per ZIP to `biomarker_tmp_outputs/`.

### 8) 8A_bmi_smok_bp.py (+ 8B SLURM)
- Builds smoking status (CPRD priority, HES fallback), derives BMI (recorded then fallback from weight/height), and merges BP (SBP/DBP).
- Output: `Enriched_Aurum_with_Biomarkers.csv`.

### 9) 9A_tests_chunks.py (+ 9B/9C SLURM)
- Per-ZIP extraction for HDL, LDL, triglycerides, HbA1c (Excel tabs) and Total Cholesterol (TXT).
- Saves variable-specific gz chunks with units when present.

### 10) 10A_tests_datacleaning.py (+ 10B SLURM)
- Standardises units (e.g., mg/dL→mmol/L) to mimic GOLD’s “unit=96 mmol/L” stream.
- Caps plausible ranges, selects result closest to index date, merges into cohort.
- Output: `FINAL_Aurum_with_Tests.csv`.

## Configuration & paths

Paths are hard-coded at the top of scripts (input globs, output dirs, code files).

For portability, mirror the GOLD layout and/or move paths to a YAML config and load with `pyyaml`.

## Performance notes

All “append/extract” scripts process ZIP members directly and/or chunk large files.

SLURM templates (`*SLURM*.sbatch`) and wrappers (`*wrapper*.sh`) throttle array size to stay under QOS limits.

Increase chunk sizes or CPU/memory in SLURM scripts as needed.

## Troubleshooting

No files found → check your glob patterns and mount permissions.

Missing columns → some exports rename fields; print `df.columns` and adjust the script’s `usecols` / renames.

Excel codelists → many exports have banner rows; scripts try multiple header rows; confirm sheet names match.

Units → tests cleaner converts mg/dL→mmol/L and infers units when missing (see comments in `10A_tests_datacleaning.py`).

Array jobs → edit placeholders (e.g., `__ARRAY_RANGE_AND_THROTTLE__`) before submitting; review logs paths set in the sbatch files.

Enriched_Aurum_with_Biomarkers.csv (baseline + smoking + BMI + BP)
