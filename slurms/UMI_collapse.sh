#!/bin/bash
#SBATCH --job-name=UMI_collapse
#SBATCH --output=UMI_collapse.out
#SBATCH --error=UMI_collapse.err
#SBATCH --time=12:00:00
#SBATCH --mem=10G
#SBATCH --partition=campus-new
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=yhuang5@fredhutch.org

python UMI_collapse.py chunks_output_new/merged_output.csv chunks_output_new/collapsed.csv chunks_output_new/collapsed.png
