#!/bin/bash

# Your SBATCH parameters here


DIR="../test"

python find_interactions.py ${DIR}/collapsed.csv 100 ${DIR}/interaction.csv ${DIR}/barcode_map.json ${DIR}/bead_distirbution.png
