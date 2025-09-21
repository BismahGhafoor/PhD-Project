import pandas as pd
import zipfile

# ----------------------------------------------------------------------
# Helper: read a tab-delimited .txt from a .zip (first .txt by default)
# ----------------------------------------------------------------------
def read_txt_from_zip(zip_path, inner_filename=None, **read_csv_kwargs):
    with zipfile.ZipFile(zip_path) as zf:
        if inner_filename is None:
            # pick the first .txt file inside the zip
            txt_members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not txt_members:
                raise FileNotFoundError(f"No .txt found inside {zip_path}")
            inner_filename = txt_members[0]
        if inner_filename not in zf.namelist():
            raise FileNotFoundError(
                f"'{inner_filename}' not found inside {zip_path}. "
                f"Available: {zf.namelist()}"
            )
        with zf.open(inner_filename) as fh:
            return pd.read_csv(fh, sep="\t", **read_csv_kwargs)

# ----------------------------------------------------------------------
# 1) Baseline file
# ----------------------------------------------------------------------
df_baseline = pd.read_csv(
    "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/baseline_grouped_df_WithNA.txt",
    sep="\t"
)

# ----------------------------------------------------------------------
# 2) Patient file (gender & yob) — now zipped
# ----------------------------------------------------------------------
patient_zip_path = (
    "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/"
    "FZ_GOLD_All_Extract_Patient_001.zip"
)

# If you know the exact inner file name, set it via inner_filename="FZ_GOLD_All_Extract_Patient_001.txt"
df_patient = read_txt_from_zip(
    patient_zip_path,
    inner_filename=None,                # or e.g. "FZ_GOLD_All_Extract_Patient_001.txt"
    usecols=["patid", "gender", "yob"],
    low_memory=False
)

# ----------------------------------------------------------------------
# 3) Ethnicity file (gen_ethnicity)
# ----------------------------------------------------------------------
df_ethnicity = pd.read_csv(
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/hes_patient_23_002869_DM.txt",
    sep="\t"
)

# ----------------------------------------------------------------------
# 4) IMD file (column name is 'e2019_imd_10')
# ----------------------------------------------------------------------
df_imd = pd.read_csv(
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/patient_2019_imd_23_002869.txt",
    sep="\t"
)

# ----------------------------------------------------------------------
# 5) Date of Death file (column name is 'dod')
# ----------------------------------------------------------------------
df_death = pd.read_csv(
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/death_patient_23_002869_DM.txt",
    sep="\t",
    low_memory=False
)

# (Optional but recommended) ensure 'patid' dtypes match to avoid merge hiccups
# for d in (df_baseline, df_patient, df_ethnicity, df_imd, df_death):
#     d["patid"] = pd.to_numeric(d["patid"], errors="raise")

# ----------------------------------------------------------------------
# Merge step by step, keeping only baseline patids
# ----------------------------------------------------------------------
df_merged = df_baseline.merge(
    df_patient[["patid", "gender", "yob"]],
    on="patid",
    how="left"
).merge(
    df_ethnicity[["patid", "gen_ethnicity"]],
    on="patid",
    how="left"
).merge(
    df_imd[["patid", "e2019_imd_10"]],
    on="patid",
    how="left"
).merge(
    df_death[["patid", "dod"]],
    on="patid",
    how="left"
)

# ----------------------------------------------------------------------
# Save the final merged DataFrame
# ----------------------------------------------------------------------
df_merged.to_csv(
    "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/baseline_with_all_features.txt",
    sep="\t",
    index=False
)
