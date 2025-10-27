import timeit
import sys
import csv
import igraph as ig
from scipy.sparse import csr_matrix, coo_matrix, tril, triu, diags
import pandas as pd
import numpy as np
import math
import ast
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)
print = partial(print, flush=True)
import datetime

#from scipy.sparse import csr_matrix, tril, triu, diags



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


def create_sparse_matrix_from_file(file_path, r1='R1_full_bc', r2='R2_full_bc', count='count'):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, sep="\t", header=None)
        r1=0
        r2=1
        count=2

    print(df.head())
    
    unique = np.unique(list(df[r1].values)+list(df[r2].values))
    mapping = dict(zip(unique,list(range(len(unique)))))

    df[r1] = df[r1].map(mapping)
    df[r2] = df[r2].map(mapping)
    
    # Convert columns to integers
    df[[r1, r2, count]] = df[[r1, r2, count]].astype(int)
    
    # Extract row indices, column indices, and values
    row_indices = df[r1].to_numpy()
    col_indices = df[r2].to_numpy()
    values = df[count].to_numpy()
    
    # Determine the shape of the matrix
    #nrows = row_indices.max() + 1
    #ncols = col_indices.max() + 1
    len_unique = len(unique)

    sp = csr_matrix((values, (row_indices, col_indices)), shape=(len_unique, len_unique))
    # Create and return the sparse matrix
    return sp, unique


def process_batch(G, nodes_batch, threshold, resolution):
    batch_results = []
    for node in nodes_batch:
        #if len(G.neighbors(node)) > threshold:
        neighbors = G.neighbors(node)
        subgraph = G.induced_subgraph(neighbors)
        clusters = subgraph.community_leiden(objective_function="modularity", resolution=resolution)
        if len(clusters) > 1:
            sorted_clusters = sorted(clusters, key=len, reverse=True)
            if len(sorted_clusters[0]) < 4*len(sorted_clusters[1]):
                batch_results.append(node)
    return batch_results


def doublet_detection_batch_parallel(G, threshold, resolution, n_jobs=-1, batch_size=100):
    # Record the start time
    start_time = timeit.default_timer()
    
    nodes = G.vs.indices
    print("Number of nodes: {}".format(len(nodes)))
    eligible_nodes = [node for node in nodes if len(G.neighbors(node)) > threshold]
    print("Number of nodes above threshold: {}".format(len(eligible_nodes)))
    
    
    # Split nodes into batches
    node_batches = [eligible_nodes[i:i + batch_size] for i in range(0, len(eligible_nodes), batch_size)]
    
    # Process batches in parallel with progress bar
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(process_batch)(G, batch, threshold, resolution) 
        for batch in tqdm(node_batches, desc="Processing batches")
    )
    
    # Flatten the list of lists to get all doublets
    doublets = [node for batch_result in results for node in batch_result]
    
    # Record the end time
    end_time = timeit.default_timer()
    
    # Calculate the runtime
    runtime = end_time - start_time
    
    print(f"Runtime of the function: {datetime.timedelta(seconds=runtime)}")
    
    return doublets


def doublet_detection_grid_resolution(G, searchspace, threshold):
    all_lens = [len(G.neighbors(x)) for x in G.vs.indices]
    #threshold = get_highest_50_percent_cutoff(all_lens)
    expected_num = expected_doublets_number(len(G.vs.indices), N=96 ** 4)
    print(f"Expected number of doublets: {expected_num}")

    all_doublets = []
    
    for i in searchspace:
        print(f"Computing resolution {i}...")
        all_doublets.append(doublet_detection_batch_parallel(G, threshold, i, n_jobs=20, batch_size=500))

    for i in range(len(searchspace)):
        print(f"Number of doublets detected at resolution {searchspace[i]}: {len(all_doublets[i])}")
        
    # Find the index of the list whose length is closest to the expected number of doublets
    index = min(range(len(all_doublets)), key=lambda i: abs(len(all_doublets[i]) - expected_num))
    
    print(f"Best resolution: {searchspace[index]}")
    print(f"Number of doublets detected: {len(all_doublets[index])}")
    
    return all_doublets[index]



if __name__=="__main__":
    start_time = timeit.default_timer()
    print("Reading data...")
    df = pd.read_csv(sys.argv[1])
    print(df.head())
    #counts_sp = create_sparse_matrix(df)
    counts_sp, unique = create_sparse_matrix_from_file(sys.argv[1])

    print("Normalizing count matrix...")
    rowsums = np.array(counts_sp.sum(axis=1)).flatten()
    colsums = np.array(counts_sp.sum(axis=0)).flatten()
    sums = rowsums+colsums
    
    r,c = counts_sp.nonzero()
    rD_sp = csr_matrix(((1.0/sums)[r], (r,c)), shape=(counts_sp.shape))
    counts_sp = counts_sp.multiply(rD_sp)
    #counts_sp = (counts_sp + counts_sp.T)/2
    #counts_sp = csr_matrix(counts_sp)
    
    #######
    '''
    # Memory-efficient normalization
    data = counts_sp.data.copy()
    # Determine row indices for each nonzero element
    row_indices = np.zeros_like(counts_sp.indices, dtype=np.int32)
    for i in range(counts_sp.shape[0]):
        row_indices[counts_sp.indptr[i]:counts_sp.indptr[i+1]] = i
    
    # Now apply normalization using correct row indices
    for i in range(len(data)):
        data[i] = data[i] / sums[row_indices[i]]
    counts_sp.data = data
    '''
    #print(counts_sp[:10,:][:,:10].toarray())
    
    def symmetrize_sparse(mat):
        """Symmetrize a sparse matrix without creating a full intermediate matrix."""
        # Get upper and lower triangular parts
        lower = tril(mat, -1)
        upper = triu(mat, 1)
        # Transpose lower and add to upper
        diag = diags(mat.diagonal())
        return diag + upper + lower.T
    
    counts_sp = symmetrize_sparse(counts_sp)
    #print(counts_sp[:10,:][:,:10].toarray())
    counts_sp = csr_matrix(counts_sp)
    #######

    print("Creating graph...")
    rows, cols = counts_sp.nonzero()
    weights = counts_sp.data
    # Create an edge list
    edges = list(zip(rows, cols))
    G = ig.Graph(n=counts_sp.shape[0], directed=False)
    G.add_edges(edges)
    G.es["weight"] = weights
    
    #G = ig.Graph.Weighted_Adjacency(counts_sp, mode=ig.ADJ_UNDIRECTED)
    all_lens = [len(G.neighbors(x)) for x in G.vs.indices]
    threshold = get_highest_50_percent_cutoff(all_lens)

    searchspace = ast.literal_eval(sys.argv[2])
    print(searchspace)
    
    print("Running doublet detection...")
    doublets_detected = doublet_detection_grid_resolution(G, searchspace, threshold)
    print(doublets_detected)
    bead_doublets = unique[doublets_detected]
    print(bead_doublets)
    end_time = timeit.default_timer()
    
    # Calculate the runtime
    runtime = end_time - start_time
    print(f"Total runtime: {datetime.timedelta(seconds=runtime)}")

    
    # Remove from input
    df_filtered = df[~df['R1_full_bc'].isin(bead_doublets) & ~df['R2_full_bc'].isin(bead_doublets)]
    

    # save file
    df_filtered.to_csv(sys.argv[3], index=False)
    

    """
    sys.argv[1]: path to the interaction sparse matrix .csv
    sys.argv[2]: search space for resolutions
    sys.argv[3]: path to output file .csv
    """



    