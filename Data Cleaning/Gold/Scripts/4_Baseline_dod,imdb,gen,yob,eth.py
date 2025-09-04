import pandas as pd

# ------------------------------------------------------------------------------
# 1) Baseline file
# ------------------------------------------------------------------------------
df_baseline = pd.read_csv(
    "/rfs/LRWE_Proj88/bg205/DataAnalysis/Baseline_dataframe/Cleaned_baselinedata/baseline_grouped_df_WithNA.txt",
    sep="\t"
)

# ------------------------------------------------------------------------------
# 2) Patient file (gender & yob)
# ------------------------------------------------------------------------------
df_patient = pd.read_csv(
    "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/GOLD/FZ_GOLD_All_Extract_Patient_001.txt",
    sep="\t"
)

# ------------------------------------------------------------------------------
# 3) Ethnicity file (gen_ethnicity)
# ------------------------------------------------------------------------------
df_ethnicity = pd.read_csv(
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/hes_patient_23_002869_DM.txt",
    sep="\t"
)

# ------------------------------------------------------------------------------
# 4) IMD file (column name is 'e2019_imd_10')
# ------------------------------------------------------------------------------
df_imd = pd.read_csv(
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/patient_2019_imd_23_002869.txt",
    sep="\t"
)

# ------------------------------------------------------------------------------
# 5) Date of Death file (column name is 'dod')
# ------------------------------------------------------------------------------
df_death = pd.read_csv(
    "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/GOLD_linked/death_patient_23_002869_DM.txt",
    sep="\t",
    low_memory=False
)

# ------------------------------------------------------------------------------
# Merge step by step, always doing how='left' to keep only baseline patids
# ------------------------------------------------------------------------------
df_merged = df_baseline.merge(
    df_patient[['patid', 'gender', 'yob']],  # from the sample you gave
    on='patid',
    how='left'
)

df_merged = df_merged.merge(
    df_ethnicity[['patid', 'gen_ethnicity']], 
    on='patid',
    how='left'
)

# Merge IMD column
df_merged = df_merged.merge(
    df_imd[['patid', 'e2019_imd_10']], 
    on='patid',
    how='left'
)

# Merge DOD column
df_merged = df_merged.merge(
    df_death[['patid', 'dod']],
    on='patid',
    how='left'
)

# ------------------------------------------------------------------------------
# Save the final merged DataFrame
# ------------------------------------------------------------------------------
df_merged.to_csv(
    "/rfs/LRWE_Proj88/bg205/DataAnalysis/Baseline_dataframe/Cleaned_baselinedata/"
    "baseline_with_all_features.txt",
    sep="\t",
    index=False
)
