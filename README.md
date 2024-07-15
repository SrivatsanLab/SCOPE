# OFST
This is the repository of Optics Free Spatial Transcriptomics (OFST). In OFST we use proximity interactions between the DNA barcodes to computationally infer their relative positions on a 2D surface. The basic idea is, if two beads (barcodes) are closed to each other, they would have more moleculars interactions, reflected by more sender receiver (SR) reactions.

![Example Image](https://github.com/SrivatsanLab/OFST/blob/main/OFST-diagram.png?raw=true)

In OFST workflow, we start with sequencing reads of both RNA sequences and SR sequences. All SR reads sequenced are attached to the unique barcodes of the beads they belong to. Thus, each pair of reads sequenced indicates an interaction between two barcodes. Because sequencing errors are supposed to exist in both SRs and barcodes, the sequencing reads are input into a preprocessing pipeline (see below) which perform error-mapping allowing sustitutions in sequences. This would output a barcode interaction matrix recording the counts of interactions between all pairs of barcodes. Next, we perform doublets detection on the interaction matrix. We perform this step because when a large number of beads present in the array, a non-negligible number of beads are supposed to be doublets, which share the same barcodes with others. After detecting the set of doublets, we either removed them or split them into separate beads.

The reconstruction of beads location starts with training a distance predictor on the input bead interaction data. Specifically, the bead interaction matrix obtained in the last step is fed into a dispersion simulator, which simulates some beads interaction data of the same distirbution, with ground truth locations of beads. We then train a random forest regressor on the simulated data, which is used to predict a distance matrix of the real data based on the interaction matrix. The next step is to cluster beads based on the interaction matrix and perform t-SNE onthe distance matrix for each single cluster to perform location reconstruction. Finally, all clusters of beads are stitched together through beads on the boundaries.

After the locations of beads are determined through reconstruction, the RNA sequences containing barcodes that map them to the beads can be assigned to locations in the array and used for downstream spatial analysis.

### Setting up the conda environment
Dependencies of running OFST are listed bellow:
   - python
   - scikit-learn
   - ......
   
To directly create the conda environment for running OFST, run `conda env create -f ofst.yml`
   

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
