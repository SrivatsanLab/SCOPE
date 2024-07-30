import time
import sys
import csv
import igraph as ig
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import math


def expected_doublets_number(n, N=96 ** 4):
    # Probability that a cell has a unique barcode
    P_unique_for_1_cell = (1 - 1/N)**(n-1)
    
    # Proportion of cells with collisions
    proportion_collision_cells = 1 - P_unique_for_1_cell
    proportion_doublets = proportion_collision_cells/2
    
    expected_number = proportion_doublets*n

    return math.ceil(expected_number)


def get_highest_50_percent_cutoff(values):
    # Sort the list in ascending order
    sorted_values = sorted(values)
    
    # Calculate the index for the 50th percentile
    cutoff_index = int(np.ceil(0.5 * len(sorted_values))) - 1
    
    # Extract the cutoff value
    cutoff_value = sorted_values[cutoff_index]
    
    return cutoff_value


def create_sparse_matrix(df):
    try:       
        # Convert columns to integers
        df[['R1_full_bc', 'R2_full_bc', 'count']] = df[['R1_full_bc', 
                                                          'R2_full_bc', 'count']].astype(int)
        
        # Extract row indices, column indices, and values
        row_indices = df['R1_full_bc'].to_numpy()
        col_indices = df['R2_full_bc'].to_numpy()
        values = df['count'].to_numpy()
        
        # Determine the shape of the matrix
        nrows = row_indices.max() + 1
        ncols = col_indices.max() + 1

        sp = csr_matrix((values, (row_indices, col_indices)), shape=(nrows, ncols))

        # Create and return the sparse matrix
        return sp
    except:
        return None


# Define a function for doublet detection
def doublet_detection(G, threshold, resolution):
    # Record the start time
    start_time = time.time()
    
    doublets = []
    nodes = G.vs.indices
    for node in nodes:
        if len(G.neighbors(node)) > threshold:
            neighbors = G.neighbors(node)
            subgraph = G.induced_subgraph(neighbors)
            clusters = subgraph.community_leiden(objective_function="modularity", resolution=resolution)
            if len(clusters) > 1:
                sorted_clusters = sorted(clusters, key=len, reverse=True)
                if len(sorted_clusters[0]) < 4*len(sorted_clusters[1]):
                    doublets.append(node)
                    
    # Record the end time
    end_time = time.time()
    
    # Calculate the runtime
    runtime = end_time - start_time
    
    print(f"Runtime of the function: {runtime:.4f} seconds")
    
    return doublets


def doublet_detection_grid_resolution(G, searchspace):
    all_lens = [len(G.neighbors(x)) for x in G.vs.indices]
    threshold = get_highest_50_percent_cutoff(all_lens)
    expected_num = expected_doublets_number(len(G.vs.indices), N=96 ** 4)
    all_doublets = []

    for i in searchspace:
        all_doublets.append(doublet_detection(G, threshold, i))
        
    # Find the index of the list whose length is closest to the expected number of doublets
    index = min(range(len(all_doublets)), key=lambda i: abs(len(all_doublets[i]) - expected_num))
    
    print(f"expected number of doublets: {expected_num}")
    print(f"best resolution: {searchspace[index]}")
    print(f"number of doublets detected: {len(all_doublets[index])}")
    
    return all_doublets[index]

df = pd.read_csv(sys.argv[1])

counts_sp = create_sparse_matrix(df)

G = ig.Graph.Weighted_Adjacency(counts_sp, mode=ig.ADJ_UNDIRECTED)

searchspace = sys.argv[2]

doublets_detected = doublet_detection_grid_resolution(G, searchspace)

# Remove from input
df_filtered = df[~df['R1_full_bc'].isin(doublets_detected) & ~df['R2_full_bc'].isin(doublets_detected)]

# save file
df_filtered.to_csv(sys.argv[3], index=False)


"""
sys.argv[1]: path to the ineracrion sparse matrix .csv
sys.argv[2]: search space for resolutions
sys.argv[3]: path to output file .csv
"""
