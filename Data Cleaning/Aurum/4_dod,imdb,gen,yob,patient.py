#!/usr/bin/env python3
import pandas as pd
import zipfile
import os
from glob import glob

# — Paths —
patient_dir        = "/rfs/LRWE_Proj88/Shared/CPRD_Raw_Data_Extract_15.01.2024/Aurum/Patient"
obs_dir            = "/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM/filtered_aurum_chunks"
gender_lookup_zip  = "/rfs/LRWE_Proj88/Shared/Lookup/202205_Lookups_CPRDAurum.zip"
hes_eth_path       = "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/Aurum_linked/hes_patient_23_002869_DM.txt"
imd_path           = "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/Aurum_linked/patient_2019_imd_23_002869.txt"
death_path         = "/rfs/LRWE_Proj88/Shared/Linkage_Raw_Data_14.02.2024/Results_type2_23_002869/Aurum_linked/death_patient_23_002869_DM.txt"
baseline_path      = "/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM/aurum_baseline_grouped_df_NoNA.txt"
codes_dir          = "/rfs/LRWE_Proj88/bg205/Codes/ethnicity_CSV_exports"

# 0. Load baseline
baseline_df = pd.read_csv(baseline_path, sep="\t", dtype={"patid": str})

# 1. Extract gender & yob from Patient ZIPs
zipped_files = sorted(glob(os.path.join(patient_dir, "*.zip")))
patient_dfs = []
for zf in zipped_files:
    with zipfile.ZipFile(zf) as z:
        txt_file = next(f for f in z.namelist() if f.endswith(".txt"))
        with z.open(txt_file) as f:
            df = pd.read_csv(f, sep="\t", usecols=["patid", "gender", "yob"], dtype=str)
            patient_dfs.append(df)
patient_all = pd.concat(patient_dfs, ignore_index=True)

# 2. Decode gender using lookup ZIP
with zipfile.ZipFile(gender_lookup_zip) as z:
    gender_file = next(f for f in z.namelist() if f.lower().endswith("gender.txt"))
    with z.open(gender_file) as f:
        gender_map = pd.read_csv(f, sep="\t", dtype=str)
patient_all = (
    patient_all
      .merge(gender_map, left_on="gender", right_on="genderid", how="left")
      .drop(columns=["genderid", "gender"])
      .rename(columns={"Description": "gender"})
)

# 3a) Primary HES ethnicity
hes_eth = pd.read_csv(
    hes_eth_path, sep="\t",
    usecols=["patid","gen_ethnicity"],
    dtype={"patid": str}
)

# 3b) Build medcode→ethnicity map from your 5 CSVs
wanted = {"Black","Missing","Other_Mixed","South_Asian","White"}
eth_map_list = []
for fp in glob(os.path.join(codes_dir, "*.csv")):
    name = os.path.splitext(os.path.basename(fp))[0]
    if name not in wanted:
        continue
    # read the real header on row-2
    df = pd.read_csv(fp, header=2, usecols=["medcodeid"], dtype={"medcodeid": str})
    df["gen_ethnicity"] = name.replace("_"," ")
    eth_map_list.append(df)

if not eth_map_list:
    raise RuntimeError("No ethnicity CSVs loaded—check your codes_dir")

ethnicity_map = pd.concat(eth_map_list, ignore_index=True)

from glob import glob

# 3c) Pull observation‐level events from your filtered_aurum_chunks (only .txt)
obs_paths = sorted(glob(os.path.join(obs_dir, "*.txt")))
obs_chunks = []

for fp in obs_paths:
    for chunk in pd.read_csv(
        fp,
        sep="\t",
        usecols=["patid", "medcodeid", "obsdate"],
        dtype={"patid": str, "medcodeid": str},
        parse_dates=["obsdate"],
        dayfirst=True,
        chunksize=200_000
    ):
        # Optional: restrict to your cohort
        # chunk = chunk[chunk["patid"].isin(baseline_df["patid"])]
        obs_chunks.append(chunk)

if not obs_chunks:
    raise RuntimeError("No Observation data loaded — check obs_dir/*.txt")

obs_all = pd.concat(obs_chunks, ignore_index=True)

# 3d) Filter to your ethnicity medcodes, pick the first record per patient
obs_eth = (
    obs_all[obs_all.medcodeid.isin(ethnicity_map.medcodeid)]
      .merge(ethnicity_map, on="medcodeid", how="left")
      .sort_values("obsdate")
      .drop_duplicates("patid", keep="first")
      .loc[:, ["patid", "gen_ethnicity"]]
)

# 3e) Combine HES + CPRD fallback (HES wins)
eth_combined = (
    hes_eth.set_index("patid")
      .combine_first(obs_eth.set_index("patid"))
      .reset_index()
)


# 3d) Filter to your ethnicity medcodes, pick first record per patient
obs_eth = (
    obs_all[obs_all.medcodeid.isin(ethnicity_map.medcodeid)]
      .merge(ethnicity_map, on="medcodeid", how="left")
      .sort_values("obsdate")
      .drop_duplicates("patid", keep="first")
      .loc[:, ["patid","gen_ethnicity"]]
)

# 3e) Combine HES + CPRD fallback (HES takes precedence)
eth_combined = (
    hes_eth.set_index("patid")
      .combine_first(obs_eth.set_index("patid"))
      .reset_index()
)

# Now merge eth_combined into your demographics and then into your baseline as before.

# 4. Extract IMD
imd = pd.read_csv(imd_path, sep="\t", usecols=["patid", "e2019_imd_10"], dtype={"patid": str})

# 5. Extract Date of Death
death_hes = pd.read_csv(death_path, sep="\t", usecols=["patid", "dod"], dtype={"patid": str})
# EMIS death date from Patient ZIPs
emis_dfs = []
for zf in zipped_files:
    with zipfile.ZipFile(zf) as z:
        txt_file = next(f for f in z.namelist() if f.endswith(".txt"))
        with z.open(txt_file) as f:
            df = pd.read_csv(f, sep="\t", usecols=["patid", "emis_ddate"], dtype=str)
            emis_dfs.append(df)
emis_df = pd.concat(emis_dfs, ignore_index=True)
emis_df = emis_df.dropna(subset=["emis_ddate"]).rename(columns={"emis_ddate": "dod"})
# Combine HES + EMIS death (HES wins)
death_combined = (
    death_hes.set_index("patid")
      .combine_first(emis_df.set_index("patid"))
      .reset_index()
)

# 6. Merge all demographics
demographics = (
    patient_all
      .merge(eth_combined, on="patid", how="left")
      .merge(imd, on="patid", how="left")
      .merge(death_combined, on="patid", how="left")
)

# 7. Enrich baseline and save
enriched = baseline_df.merge(demographics, on="patid", how="left")
output_path = "/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM/Enriched_baseline_with_demographics.csv"
enriched.to_csv(output_path, index=False)
print("Enriched baseline saved to:", output_path)
