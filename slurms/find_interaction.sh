#!/bin/bash
#SBATCH --job-name=find_interaction
#SBATCH --output=find_interaction.out
#SBATCH --error=find_interaction.err
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --partition=campus-new
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=yhuang5@fredhutch.org

python find_interactions.py chunks_output/collapsed.csv 100 chunks_output/interaction.csv chunks_output/merged_dict.csv chunks_output/mapped_dict.csv chunks_output/barcode_map.json chunks_output/bead_distirbution.png