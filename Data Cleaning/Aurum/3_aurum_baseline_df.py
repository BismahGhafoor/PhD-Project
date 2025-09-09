import pandas as pd
import glob
import os

# =============================================================================
# Step 1: Backup chunk files
# =============================================================================
chunk_dir = "/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM/filtered_aurum_chunks"
chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "Cleaned_AURUM_Observation_*.txt")))
backup_dir = "aurum_backup_chunks"

if not chunk_files:
    raise FileNotFoundError(f"No files found in {chunk_dir} matching pattern 'Cleaned_AURUM_Observation_*.txt'")

if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

for file in chunk_files:
    backup_file = os.path.join(backup_dir, os.path.basename(file))
    if not os.path.exists(backup_file):
        os.rename(file, backup_file)
        print(f"Backed up {file} to {backup_file}")

# =============================================================================
# Step 2: Process each chunk file
# =============================================================================
def process_chunk(file):
    df = pd.read_csv(file, sep="\t", dtype=str)
    df = df[["patid", "obsdate", "medcodeid"]]
    df["obsdate"] = pd.to_datetime(df["obsdate"], errors='coerce', dayfirst=True)
    df = df.dropna(subset=["obsdate"])
    return df

all_data = pd.concat([process_chunk(file) for file in chunk_files], ignore_index=True)

# =============================================================================
# Step 3: Map medcodeids to diabetes types
# =============================================================================
codes_file = "/rfs/LRWE_Proj88/bg205/DataAnalysis/Data_Cleaning_AURUM/filtered_diabetes_AURUM_codes.txt"
codes_df = pd.read_csv(codes_file, sep="\t", dtype=str)
medcode_to_type = codes_df.set_index("medcodeid")["type"].to_dict()
all_data["diabetes_type"] = all_data["medcodeid"].map(medcode_to_type)

# Drop rows where diabetes_type is not 1 or 2
all_data = all_data[all_data["diabetes_type"].isin(["1", "2"])]

# =============================================================================
# Step 4: Save ungrouped baseline DataFrame (before deduplication)
# =============================================================================
all_data.to_csv("aurum_baseline_ungrouped_df_NoNA.txt", sep="\t", index=False)
print("Ungrouped DataFrame saved as 'aurum_baseline_ungrouped_df_NoNA.txt'")

# =============================================================================
# Step 5: Derive indexdate per patient (earliest diagnosis)
# =============================================================================
grouped_df = (
    all_data.sort_values(by=["patid", "obsdate"])
    .drop_duplicates(subset="patid", keep="first")
    .copy()
)
grouped_df["indexdate"] = grouped_df["obsdate"]

# Save grouped DataFrame
grouped_df.to_csv("aurum_baseline_grouped_df_NoNA.txt", sep="\t", index=False)
print("Grouped DataFrame saved as 'aurum_baseline_grouped_df_NoNA.txt'")

# =============================================================================
# Step 6: Save separate files for Type 1 and Type 2
# =============================================================================
type_1_df = grouped_df[grouped_df["diabetes_type"] == "1"]
type_2_df = grouped_df[grouped_df["diabetes_type"] == "2"]

print("\nSummary of Diabetes Types:")
print(f"Total Type 1 Diabetes patients: {type_1_df['patid'].nunique()}")
print(f"Total Type 2 Diabetes patients: {type_2_df['patid'].nunique()}")

type_1_df.to_csv("aurum_baseline_Type_1_Diabetes_NoNA.txt", sep="\t", index=False)
type_2_df.to_csv("aurum_baseline_Type_2_Diabetes_NoNA.txt", sep="\t", index=False)
