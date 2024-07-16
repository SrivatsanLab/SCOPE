#!/bin/bash
#SBATCH --job-name=split_files
#SBATCH --output=split_files.out
#SBATCH --error=split_files.err
#SBATCH --ntasks=2
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --partition=campus-new
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=yhuang5@fredhutch.org

# Define input files and parameters
R1_INPUT="R1.fastq.gz"
R2_INPUT="R2.fastq.gz"
OUTPUT_PREFIX_R1="output_R1/R1_split"
OUTPUT_PREFIX_R2="output_R2/R2_split"
NUM_CHUNKS=5  # Set the number of chunks you want

# Create output directories if they don't exist
mkdir -p output_R1
mkdir -p output_R2

# Split R1 and R2 in parallel
srun -N1 -n1 python split_fastq.py $R1_INPUT $OUTPUT_PREFIX_R1 $NUM_CHUNKS &
srun -N1 -n1 python split_fastq.py $R2_INPUT $OUTPUT_PREFIX_R2 $NUM_CHUNKS &
wait
