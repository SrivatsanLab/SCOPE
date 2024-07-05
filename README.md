# OFST
Optics Free Spatial Transcriptomics Repository

![Example Image](https://raw.githubusercontent.com/SrivatsanLab/OFST/main/OFST-diagram.png?token=GHSAT0AAAAAACTRUBB2WENIJE3XM4SDEVNSZUICJBQ)

### Error Correction & Creating an interaction matrix

1. **Run `extract_error_correct_sender_receiver.py`**

   ```sh
   nohup python extract_error_correct_sender_receiver.py R1.fastq.gz R2.fastq.gz corrected.txt corrected.csv > script_output.log 2>&1 &
2. **Run `UMI_collapse.py`**

   ```sh
   python UMI_collapse.py corrected.csv collapsed.csv
3. **Run `find_interactions.py`**

   ```sh
   python find_interactions.py collapsed.csv index.json subset.csv numerical.txt interaction.txt

### Doublets Detection
1. **Run `doublet_detection.py`**

   ```sh
   python doublet_detection.py interaction_graph.graphml 1000 doublets.csv

### Reconstruction of Beads Location
1. **Run `optics-free.py`**

   ```sh
   python optics-free.py -s sample_id -i interaction.csv
