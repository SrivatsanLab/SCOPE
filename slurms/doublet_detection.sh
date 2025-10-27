#!/bin/bash

# Your SBATCH parameters here
#SBATCH --cpus-per-task=24

DIR="test"

python doublet_detection.py ${DIR}/interaction.csv "[0.05, 0.1, 0.2]" ${DIR}/filtered_interaction.csv
