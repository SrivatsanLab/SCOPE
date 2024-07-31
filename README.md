# OFST
This is the repository of Optics Free Spatial Transcriptomics (OFST). In OFST we use proximity interactions between the DNA barcodes to computationally infer their relative positions on a 2D surface. The basic idea is, if two beads (barcodes) are closed to each other, they would have more moleculars interactions, reflected by more sender receiver (SR) reactions.

![Example Image](https://github.com/SrivatsanLab/OFST/blob/main/OFST-diagram.png?raw=true)

In OFST workflow, we start with sequencing reads of both RNA sequences and SR sequences. All SR reads sequenced are attached to the unique barcodes of the beads they belong to. Thus, each pair of reads sequenced indicates an interaction between two barcodes. Because sequencing errors are supposed to exist in both SRs and barcodes, the sequencing reads are input into a preprocessing pipeline (see below) which perform error-mapping allowing sustitutions in sequences. This would output a barcode interaction matrix recording the counts of interactions between all pairs of barcodes. Next, we perform doublets detection on the interaction matrix. We perform this step because when a large number of beads present in the array, a non-negligible number of beads are supposed to be doublets, which share the same barcodes with others. After detecting the set of doublets, we either removed them or split them into separate beads.

The reconstruction of beads location starts with training a distance predictor on the input bead interaction data. Specifically, the bead interaction matrix obtained in the last step is fed into a dispersion simulator, which simulates some beads interaction data of the same distirbution, with ground truth locations of beads. We then train a random forest regressor on the simulated data, which is used to predict a distance matrix of the real data based on the interaction matrix. The next step is to cluster beads based on the interaction matrix and perform t-SNE onthe distance matrix for each single cluster to perform location reconstruction. Finally, all clusters of beads are stitched together through beads on the boundaries.

After the locations of beads are determined through reconstruction, the RNA sequences containing barcodes that map them to the beads can be assigned to locations in the array and used for downstream spatial analysis.

### Setting up the environment
Dependencies of running OFST are listed bellow:
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

To directly create the conda environment for running OFST, run `conda env create -f ofst.yml`
   

### Running the scripts
SLURM scripts for the pipeline are provided. Here's how to run them:

1. **Run `error_correction.sh`**

   ```sh
   sbatch slurms/error_correction.sh
   ```
   This would start processing the paired of sequencing files in parallel. Specify the number of chunks in the script and it will generate the same number of files that contain mapped reads.
2. **Run `UMI_collapse.sh`**

   ```sh
   sbatch slurms/UMI_collapse.sh
   ```
   This first merges all output reads file from last step into one and then collapses identical reads which contain the same pairs of UMIs. This would output a reads file with UMI-collpased reads and a knee plot showing the distribution of barcodes interaction counts.
3. **Run `find_interaction.sh`**

   ```sh
   sbatch slurms/find_interaction.sh
   ```
   The input to this step is the UMI-collapsed reads file. Users also need to provide a threshold for filtering barcodes by their interaction counts with other barcodes. This threshold can be set by determining the knee point from the knee plot output by last step. The output of this step are a sparse matrix of barcode interactions, a dictionary that maps barcode sequences to indices and a plot of barcodes interactions of randomly selected barcodes.  
4. **Run `doublet_detection.sh`**

   ```sh
   sbatch slurms/doublet_detection.sh
   ```
   The purpose of this step is to remove doublets (pairs of beads that share the same barcodes) exsiting in the array that might potentially affect the reconstruction results. The input to this step are a sparse matrix of barcode interactions and a list indicating the search space of resolutions for performing leiden clustering; the output is a sparse matrix of bead interactions with doublets removed.
5. **Run `reconstruction.sh`**

   ```sh
   sbatch slurms/reconstruction.sh
   ```
   The input to this step is a sparse matrix of bead interactions (ideally with doublets removed). It outputs a plot showing the location reconstruction of beads and a file that contains the predicted coordinates of beads in the array.

