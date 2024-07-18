#!/bin/bash
#SBATCH --job-name=my_job         # Job name
#SBATCH --output=%x_%j.out        # Standard output and error log (%x = job name, %j = job ID)
#SBATCH --error=%x_%j.err         # Error file (%x = job name, %j = job ID)
#SBATCH --partition=campus-new # Partition name
#SBATCH --mem=8G                  # Memory per node
#SBATCH --time=12:00:00           # Time limit hrs:min:sec
#SBATCH --cpus-per-task=24        # Number of CPU cores per task
#SBATCH --mail-type=END,FAIL      # Mail events (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yhuang5@fredhutch.org  # Where to send mail

# Your job commands go here

python optics-free.py -s Asymmetric4 -i chunks_output/interaction.csv

