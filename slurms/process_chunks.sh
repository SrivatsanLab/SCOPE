#!/bin/bash
#SBATCH --job-name=process_chunks
#SBATCH --output=process_chunks.out
#SBATCH --error=process_chunks.err
#SBATCH --array=1-10  # Adjust based on the number of chunks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=4G
#SBATCH --partition=campus-new
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=yhuang5@f

# Define input and output directories
R1_INPUT="/fh/fast/srivatsan_s/grp/SrivatsanLab/Sanjay/OFST/data/20240610_NEXTSEQ_P2_200cycles_SR_cDNA_slide10_coverslip_snellen/SSSP10195_S5_R1_001.fastq.gz"
R2_INPUT="/fh/fast/srivatsan_s/grp/SrivatsanLab/Sanjay/OFST/data/20240610_NEXTSEQ_P2_200cycles_SR_cDNA_slide10_coverslip_snellen/SSSP10195_S5_R2_001.fastq.gz"
OUTPUT_DIR="chunks_output"

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

# Define the output json file for this task
OUTPUT_DICT="${OUTPUT_DIR}/dict_${TASK_ID}.csv"

# Run the process_chunk script
python process_chunk.py $R1_INPUT $R2_INPUT $OUTPUT_FILE $OUTPUT_DICT $start_sequence $num_sequences
