# Optics-Free Spatial Transcriptomics (OFST) / Spatial reConstruction via Oligonucleotide Proximity Encoding (SCOPE)
SCOPE uses proximity interactions between the DNA barcodes to computationally infer their relative positions on a 2D surface. 

The OFST/SCOPE platform is built on an array of densely packed 20µm polyacrylamide hydrogel beads. Each bead contains 100µM 5'-acrydite oligos -- a portion of which can be programmatically cleaved by an enzymatic reaction. Cleaved oligo barcodes, termed 'senders' can diffuse freely and hybridize to tethered oligo barcodes, 'receivers', on proximal beads. Cleaved senders will diffuse radially such that receiver oligos on proximal beads will have many more sender-receiver (SR) reactions than receiver oligos on distal beads. Thus, the spatial relationships between beads are reflected in the magnitude and proportion of respective SR reactions. The beads that contribute to these reactions are identified by sequencing the hybridized sender-receiver oligos. 

![Example Image](https://github.com/SrivatsanLab/OFST/blob/main/OFST-diagram.png?raw=true)

In the above workflow, we begin by sequencing RNA and SR reads. All SR reads contain the combined barcodes from the sending and receiving beads. Thus, each pair of sequenced reads represents an interaction between two beads. To account for substitutions, insertions, and deletions, sequencing reads are input into a preprocessing pipeline (see below) which performs error-mapping and decoding. This outputs a barcode interaction matrix recording the counts of interactions between all pairs of barcodes. 

Next, we perform doublet detection on the interaction matrix. Doublets occur when multiple hydrogel beads contain identical barcodes. Once doublets are detected, they are removed from the interaction matrix.

![Example Image](https://github.com/SrivatsanLab/OFST/blob/main/puzzle_solution_final.png?raw=true)

Spatial reconstruction of the bead array begins with training a distance predictor on a subset of the bead interaction data. Specifically, the bead interaction matrix obtained in the previous step is fed into a dispersion simulator which simulates bead interaction data of the same distribution along with ground truth locations of beads. We then train a random forest regressor on the simulated data, which predicts a distance matrix of the real data based on the interaction matrix. Next, we cluster beads based on the interaction matrix and perform t-SNE on the distance matrix of each cluster to perform local reconstruction. Finally, all bead clusters are stitched together using beads along the outer boundaries.

![Example Image](https://github.com/SrivatsanLab/OFST/blob/main/RNA-beads-connection.png?raw=true)

After the bead locations are reconstructed, the RNA sequences associated with each bead are assigned to spatial positions in the array and used for downstream spatial transcriptomics.

### Setting up the environment
Dependencies for OFST/SCOPE are listed below:
   - python
   - scikit-learn
   - alphashape
   - igraph
   - leidenalg
   - matplotlib
   - pandas
   - python-igraph
   - python-levenshtein
   - tqdm
   - scikit-image
   - scanpy

To create the conda environment for running OFST, run `conda env create -f ofst.yml`
   

### Running the scripts
We provide SLURM scripts for the pipeline and test data. Here's how to run them:

1. **Run `error_correction.sh`**

   ```sh
   sbatch slurms/error_correction.sh
   ```
   This would start processing the paired of sequencing files in parallel. Specify the number of chunks in the script and it will generate the same number of files that contain mapped reads.

2. **Run `UMI_collapse.sh`**

   ```sh
   sbatch slurms/UMI_collapse.sh
   ```
   This first merges all output read files from Step 1 and then collapses identical reads that contain the same pairs of UMIs. This outputs a reads file with UMI-collpased reads and a knee plot showing the distribution of barcode interaction counts.
   
4. **Run `find_interaction.sh`**

   ```sh
   sbatch slurms/find_interaction.sh
   ```
   The input to this step is the UMI-collapsed reads file. Users must provide a threshold for filtering barcodes by their interaction counts with other barcodes. This threshold can be set by determining the knee point from the knee plot output from Step 2. The output of this step includes a sparse matrix of barcode interactions, a dictionary that maps barcode sequences to indices, and a plot of interactions of randomly selected barcodes.  
5. **Run `doublet_detection.sh`** (optional)

   ```sh
   sbatch slurms/doublet_detection.sh
   ```
   This step removes doublets (pairs of beads that share the same barcodes) existing in the array that might affect the reconstruction results. The inputs for this step are a sparse matrix of barcode interactions and a list indicating the search space of resolutions for performing Leiden clustering. The output is a sparse matrix of bead interactions with doublets removed.
6. **Run `reconstruction.sh`**

   ```sh
   sbatch slurms/reconstruction.sh
   ```
   The input to this step is a sparse matrix of bead interactions (ideally with doublets removed). It outputs a plot showing the location reconstruction of beads and a file that contains the predicted coordinates of beads in the array.

