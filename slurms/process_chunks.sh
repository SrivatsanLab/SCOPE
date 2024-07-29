#!/bin/bash

# Your SBATCH parameters here
#SBATCH --array=1-10  # Adjust based on the number of chunks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Define input and output directories
R1_INPUT="../test/R1.fastq.gz"
R2_INPUT="../test/R2.fastq.gz"
OUTPUT_DIR="../test"

mkdir -p $OUTPUT_DIR

# Calculate the total number of sequences in the input files
total_sequences=$(zcat $R1_INPUT | wc -l)
total_sequences=$((total_sequences / 4))

# Calculate the number of sequences per chunk
num_chunks=10  # Set the number of chunks you want
sequences_per_chunk=$((total_sequences / num_chunks))

# Get the SLURM array task ID
TASK_ID=${SLURM_ARRAY_TASK_ID}

# Calculate the start sequence for this task
start_sequence=$(( (TASK_ID - 1) * sequences_per_chunk ))

# Calculate the number of sequences to process in this task
if [ $TASK_ID -eq $num_chunks ]; then
    # The last chunk should process all remaining sequences
    num_sequences=$(( total_sequences - start_sequence ))
else
    num_sequences=$sequences_per_chunk
fi

# Define the output file for this task
OUTPUT_FILE="${OUTPUT_DIR}/output_${TASK_ID}.csv"

# Run the process_chunk script
python process_chunk.py $R1_INPUT $R2_INPUT $OUTPUT_FILE $start_sequence $num_sequences
