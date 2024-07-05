import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
from scipy.sparse import csr_matrix

# load de-duplicated library to csv
nextera_unique = pd.read_csv(sys.argv[1])


# Calculate total counts of each barcode across both entry columns
barcode_counts = nextera_unique[["R1_full_bc", "R2_full_bc"]].stack().value_counts()


# Filter for barcodes that appear at least x times (look at kneeplot)
total_bc_cutoff = int(sys.argv[2])
print(total_bc_cutoff)
real_barcodes = barcode_counts[barcode_counts >= total_bc_cutoff].index.tolist() #used total bc counts of R1 + R2

print(f"Number of barcodes after filtering: {len(real_barcodes)}")

filtered_df = nextera_unique[nextera_unique['R1_full_bc'].isin(real_barcodes) & nextera_unique['R2_full_bc'].isin(real_barcodes)]


# Filter for barcodes that appear in both entry columns
filtered_barcodes = []

barcode_set_entry1 = set(filtered_df['R1_full_bc'])
barcode_set_entry2 = set(filtered_df['R2_full_bc'])
filtered_barcodes = list(barcode_set_entry1.intersection(barcode_set_entry2))

filtered_df = filtered_df[filtered_df['R1_full_bc'].isin(filtered_barcodes)]
filtered_df = filtered_df[filtered_df['R2_full_bc'].isin(filtered_barcodes)]


# Getting matrix of interactions
interaction_df = filtered_df.groupby(["R1_full_bc", "R2_full_bc"]).size().reset_index().rename(columns={0:'count'})

# Create a set to store unique barcodes
unique_barcodes = set()

# Mapping dictionary to store the relationship between barcode and integer
barcode_to_integer_mapping = {}

# Function to replace barcodes with unique integers and update the mapping dictionary
def replace_with_integer(barcode):
    if barcode not in unique_barcodes:
        unique_barcodes.add(barcode)
        barcode_to_integer_mapping[barcode] = len(unique_barcodes) - 1
    return barcode_to_integer_mapping[barcode]

# Apply the function to each element in the specified columns
interaction_df['R1_full_bc'] = interaction_df['R1_full_bc'].apply(replace_with_integer)
interaction_df['R2_full_bc'] = interaction_df['R2_full_bc'].apply(replace_with_integer)

# save file
interaction_df.to_csv(sys.argv[3])


# write a json file with barcode to numerical index matching

with open(sys.argv[4], "w") as f:
     f.write(json.dumps(barcode_to_integer_mapping))


# Create a sparse matrix
sparse_matrix = csr_matrix((interaction_df['count'], (interaction_df['R1_full_bc'], interaction_df['R2_full_bc'])))

# Randomly sample 18 beads for plotting
sampled_num = 18
total_num_beads = sparse_matrix.shape[0]
random_sampled_beads = np.random.choice([*range(total_num_beads)], size=sampled_num, replace=False)

fig, axs = plt.subplots(3, 6, figsize=(16, 8))

axs = axs.flatten()

for i in range(sampled_num):
    
    interactions = sparse_matrix.getrow(random_sampled_beads[i]).toarray().flatten()
    non_zero_interactions = sorted(interactions[interactions != 0])[::-1]
    
    axs[i].scatter([*range(len(non_zero_interactions))], non_zero_interactions)
    axs[i].set_title(f'Bead {random_sampled_beads[i]}')
    
plt.tight_layout()

plt.savefig(sys.argv[5], format='pdf')


"""

sys.argv[1]: deduplicated lib to load (.csv)
sys.argv[2]: threshold for filtering barcodes (int)
sys.argv[3]: interactions.csv output filename (.csv)
sys.argv[4]: output .json file with barcode to numerical index matching (.json)
sys.argv[5]: the library's barcode strings have been replaced with numerical indices (.txt file)

"""

