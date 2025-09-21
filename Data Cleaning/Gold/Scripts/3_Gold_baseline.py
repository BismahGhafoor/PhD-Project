import pandas as pd
import glob
import os

# =============================================================================
# Step 1: Backup chunk files
# =============================================================================
chunk_dir = "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/backup_chunks"
chunk_files = sorted(glob.glob(os.path.join(chunk_dir, "Cleaned_GOLD_Extract_Clinical_*.txt")))
backup_dir = "backup_chunks"

# Create backup directory if it doesn't exist
if not os.path.exists(backup_dir):
    os.makedirs(backup_dir)

# Backup files
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
    df = df[["patid", "eventdate", "medcode"]]  # Keep only relevant columns
    df["eventdate"] = pd.to_datetime(df["eventdate"], errors='coerce', dayfirst=True)  # Convert dates

    # Keep only the earliest eventdate per patient_id
    df = df.sort_values(by=["patid", "eventdate"]).drop_duplicates(subset="patid", keep="first")
    return df

# Process all chunk files and concatenate them
all_data = pd.concat([process_chunk(file) for file in chunk_files], ignore_index=True)

# =============================================================================
# Step 3: Map medcodes to diabetes types
# =============================================================================
# Load the filtered diabetes codes
codes_file = "/scratch/alice/b/bg205/DataCleaning_FINAL_Gold/filtered_diabetes_codes.txt"
codes_df = pd.read_csv(codes_file, sep="\t", dtype=str)
codes_df.rename(columns={"code": "medcode"}, inplace=True)

# Create a mapping dictionary
medcode_to_type = codes_df.set_index("medcode")["type"].to_dict()

# Map the diabetes type to the all_data DataFrame
all_data["diabetes_type"] = all_data["medcode"].map(medcode_to_type)

# =============================================================================
# Step 4: Save the baseline DataFrame
# =============================================================================
all_data.to_csv("baseline_ungrouped_df_WithNA.txt", sep="\t", index=False)
print("Baseline DataFrame created and saved as 'baseline_df_WithNA.txt'")

# =============================================================================
# Step 5: Verify the earliest eventdate for each patid
# =============================================================================
print("\nVerifying that the earliest eventdate was kept...")
missing_baseline = all_data[all_data["eventdate"].isna()]
print(f"Missing dates in baseline dataframe: {len(missing_baseline)}")

for file in sorted(glob.glob("Cleaned_GOLD_Extract_Clinical_*.txt")):
    df = pd.read_csv(file, sep="\t", dtype=str)
    df["eventdate"] = pd.to_datetime(df["eventdate"], errors="coerce", dayfirst=True)

    # Find the earliest eventdate per patid in the original file
    earliest_dates = df.groupby("patid")["eventdate"].min()

    # Merge with the baseline dataframe to compare dates
    merged = all_data.merge(earliest_dates, on="patid", suffixes=("_baseline", "_original"))

    # Check if the dates match
    mismatches = merged[merged["eventdate_baseline"] != merged["eventdate_original"]]

    if not mismatches.empty:
        print("Mismatch found!")
        print(mismatches.head(10))

        # Additional logging to diagnose issues
        print("\nDetails of mismatches:")
        for patid in mismatches["patid"].unique():
            print(f"\nPatient ID: {patid}")
            original_dates = df[df["patid"] == patid]["eventdate"].dropna().tolist()
            print(f"Original dates in chunk file: {original_dates}")
            baseline_date = mismatches[mismatches["patid"] == patid]["eventdate_baseline"].values[0]
            print(f"Baseline date: {baseline_date}")
    else:
        print(f"No mismatches in {file}. The earliest dates were correctly kept.")

# =============================================================================
# Step 6: Group Diabetes Types into Type 1 and Type 2
# =============================================================================
# Filter Type 1 and Type 2 Diabetes patients
type_1_patients = all_data[all_data["diabetes_type"] == "1"]
type_2_patients = all_data[all_data["diabetes_type"] == "2"]

# Print summary statistics
print("\nSummary of Diabetes Types (After Grouping):")
print(f"Total Type 1 Diabetes patients: {type_1_patients['patid'].nunique()}")
print(f"Total Type 2 Diabetes patients: {type_2_patients['patid'].nunique()}")

# Save the grouped DataFrames to separate files
type_1_patients.to_csv("baseline_Type_1_Diabetes_WithNA.txt", sep="\t", index=False)
type_2_patients.to_csv("baseline_Type_2_Diabetes_WithNA.txt", sep="\t", index=False)

# =============================================================================
# Step 7: Save a Combined Grouped DataFrame
# =============================================================================
# Combine both groups into a single DataFrame and save
grouped_df = pd.concat([type_1_patients, type_2_patients], ignore_index=True)
grouped_df.to_csv("baseline_grouped_df_WithNA.txt", sep="\t", index=False)

print("\nGrouped DataFrame saved as 'baseline_grouped_df_WithNA.txt'")
