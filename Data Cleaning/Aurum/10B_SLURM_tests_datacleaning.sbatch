#!/bin/bash
#SBATCH --job-name=aurum_biomarker_cleaning
#SBATCH --output=/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/aurum_cleaning_%j.out
#SBATCH --error=/scratch/alice/b/bg205/DataCleaning_FINAL_Aurum/aurum_cleaning_%j.err
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH --mem=100G          # Request all memory on node
#SBATCH --cpus-per-task=8   
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bg205@leicester.ac.uk

# Load Python module
module load python/3.10.12-tggsi7t
source ~/aurum_env/bin/activate

# Run the script
python3 -u 10A_tests_datacleaning.py
