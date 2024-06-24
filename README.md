# OFST
Optics Free Spatial Transcriptomics Repository

### Creating an interaction matrix

1. **Run `extract_error_correct_sender_receiver.py`**

   ```sh
   nohup python extract_error_correct_sender_receiver.py R1.fastq.gz R2.fastq.gz corrected.txt corrected.csv > script_output.log 2>&1 &
2. **Run `UMI_collapse.py`**

   ```sh
   python UMI_collapse.py corrected.csv collapsed.csv
3. **Run `find_interactions.py`**

   ```sh
   python find_interactions.py collapsed.csv index.json subset.csv numerical.txt interaction.txt
