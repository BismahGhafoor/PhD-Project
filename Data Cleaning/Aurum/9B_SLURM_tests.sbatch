#!/bin/bash
# This is template_smoking_job.sbatch

#SBATCH --job-name=test_batch_job         # Name for each small batch job
#SBATCH --array=__ARRAY_RANGE_AND_THROTTLE__  # << THIS IS A PLACEHOLDER the wrapper will replace
#SBATCH --time=02:00:00                      # Time PER TASK in this batch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/scratch/alice/b/bg205/smoking_run/tests_logs/test_biomarkers_%A_%a.out
#SBATCH --error=/scratch/alice/b/bg205/smoking_run/tests_logs/test_biomarkers_%A_%a.err


# --- IMPORTANT: Load your Python environment ---
module purge  # clears Python modules that conflict with Conda

# Activate Conda properly
source ~/miniconda3/etc/profile.d/conda.sh
conda activate aurum_en


# --- Define where your Python script and data are on scratch ---
SCRATCH_WORK_DIR="/scratch/alice/b/bg205/smoking_run"
PYTHON_PROCESSING_SCRIPT="8_tests.py"

# --- Go to the working directory on scratch ---
cd "${SCRATCH_WORK_DIR}" || exit 1 # Exit if 'cd' fails

# --- Echo some info for logging ---
echo "Slurm Array Job ID (for this batch): $SLURM_ARRAY_JOB_ID"
echo "Slurm Array Task ID (within this batch): $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Current working directory: $(pwd)"
echo "Executing Python script: python ${PYTHON_PROCESSING_SCRIPT} $SLURM_ARRAY_TASK_ID"

# --- Run your Python script ---
# Your Python script (process_zip_chunk.py) takes the SLURM_ARRAY_TASK_ID
# as an argument to know which ZIP file (from 0 to 1046) it needs to process.
python "${PYTHON_PROCESSING_SCRIPT}" $SLURM_ARRAY_TASK_ID

echo "Python script for Task $SLURM_ARRAY_TASK_ID finished."
