#!/bin/bash
#SBATCH --job-name=process_chunks
#SBATCH --output=process_chunks.out
#SBATCH --error=process_chunks.err
#SBATCH --array=1-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=campus-new
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=yhuang5@fredhutch.org

# Define input and output directories
CHUNK_DIR="."
OUTPUT_DIR="chunks_output"

# Get the SLURM array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Define the input files for this task
R1_FILE="${CHUNK_DIR}/R1_${TASK_ID}.fastq.gz"
R2_FILE="${CHUNK_DIR}/R2_${TASK_ID}.fastq.gz"
OUTPUT_FILE="${OUTPUT_DIR}/output_${TASK_ID}.csv"

# Run the process_chunk script
python process_chunk.py $R1_FILE $R2_FILE $OUTPUT_FILE

