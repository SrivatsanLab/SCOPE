#!/bin/bash

# Your SBATCH parameters here
#SBATCH --cpus-per-task=24       

DIR="../test"

python ../reconstruction.py -s ${DIR}/results -i ${DIR}/interaction.csv

