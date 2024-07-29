#!/bin/bash

# Your SBATCH parameters here


DIR="../test"
MERGED_OUTPUT="${DIR}/merged_output.csv"

# Merge all individual output files into a single file
cat ${DIR}/output_*.csv > $MERGED_OUTPUT

# Perform UMI collapsing on the merged output file
python ../UMI_collapse.py ${DIR}/merged_output.csv ${DIR}/collapsed.csv ${DIR}/collapsed.png
