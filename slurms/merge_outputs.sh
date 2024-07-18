#!/bin/bash
#SBATCH --job-name=merge_outputs
#SBATCH --output=merge_outputs.out
#SBATCH --error=merge_outputs.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=campus-new
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=yhuang5@fredhutch.org

# Define output directory and merged output file
MERGED_OUTPUT_DIR="chunks_output_new"
MERGED_OUTPUT="${MERGED_OUTPUT_DIR}/merged_output.csv"

# Merge all individual output files into a single file
cat ${MERGED_OUTPUT_DIR}/output_*.csv > $MERGED_OUTPUT

python merge_dict.py
