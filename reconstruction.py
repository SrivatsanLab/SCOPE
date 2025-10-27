from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#from sklearn.manifold import TSNE
#from scipy.optimize import minimize
from scipy.stats import binned_statistic
import timeit
import datetime
import matplotlib as mpl
import os
import sys
#from sklearn.manifold import SpectralEmbedding
from sklearn.ensemble import RandomForestRegressor
#import joblib
from joblib import Parallel, delayed
import igraph
import igraph as ig
import leidenalg as la
from collections import Counter, defaultdict
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
#from scipy.optimize import basinhopping

#sys.path.append('/net/shendure/vol8/projects/sanjayk/srivatsan/sci-space-v2')
#from ssv2.simulation import BaseSimulation
from simulation import BaseSimulation

from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, tril, triu, diags

from scipy.spatial import KDTree
from alphashape import alphashape
from shapely.geometry import Point

from sklearn.neighbors import BallTree

import skimage
from skimage.morphology import binary_erosion
from skimage.segmentation import watershed, chan_vese
from skimage.measure import label, regionprops

import math
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsTransformer#, KNeighborsClassifier

#from sklearn.metrics import f1_score
#from sklearn.metrics import precision_score

import argparse
import scanpy as sc
#from scipy.optimize import linear_sum_assignment

#from sklearn.cluster import SpectralClustering

from scipy.cluster.hierarchy import linkage, fcluster
#from sklearn.cluster import AgglomerativeClustering

import subprocess
import time

mpl.rcParams['figure.dpi'] = 500
np.random.seed(0)

import warnings
warnings.filterwarnings('ignore')

import functools
print = functools.partial(print, flush=True)

from shapely.geometry import Polygon, Point, LineString, MultiPolygon
from shapely.ops import unary_union

#from collections import defaultdict
from matplotlib.patches import Polygon as mplPolygon
from scipy.stats import chisquare

#from shapely.geometry import Polygon, Point
#from shapely.ops import unary_union
from scipy.spatial import Delaunay
#import shapely.errors

#from sklearn.neighbors import KernelDensity
import anndata as ad

from umap.umap_ import UMAP

import sys
#sys.path.append('../')
#import optics_free

from alphashape import alphashape

import networkx as nx

import scanpy as sc

#from memory_profiler import profile, memory_usage
#import tracemalloc


def flatten(xss):
    return [x for xs in xss for x in xs]


def downsample(counts_sp, pct=1.0):
    adata = sc.AnnData(counts_sp)
    total_sum = adata.X.sum()
    adata_ds = sc.pp.downsample_counts(adata, total_counts=pct*total_sum, copy=True)
    counts_sp = csr_matrix(adata_ds.X)
    return counts_sp


#@profile
def create_sparse_matrix_from_file(file_path, r1='R1_full_bc', r2='R2_full_bc', count='count'):
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_csv(file_path, sep="\t", header=None)
        r1=0
        r2=1
        count=2
    
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
    nrows = row_indices.max() + 1
    ncols = col_indices.max() + 1
    len_unique = len(unique)

    sp = csr_matrix((values, (row_indices, col_indices)), shape=(len_unique, len_unique), dtype=np.float32)
    # Create and return the sparse matrix
    return sp, unique


#@profile
def similarity_distance_mapping(dir, counts_sp):
    rowsums = np.array(counts_sp.sum(axis=1)).flatten()
    colsums = np.array(counts_sp.sum(axis=0)).flatten()
    sums = rowsums+colsums
    r,c = counts_sp.nonzero()
    rD_sp = csr_matrix(((1.0/sums)[r], (r,c)), shape=(counts_sp.shape), dtype=np.float32)
    counts_sp = counts_sp.multiply(rD_sp)
    #counts_sp = (counts_sp + counts_sp.T)/2
    #counts_sp = csr_matrix(counts_sp)
    del rD_sp

    def symmetrize_sparse(mat):
        """Symmetrize a sparse matrix without creating a full intermediate matrix."""
        # Get upper and lower triangular parts
        lower = tril(mat, -1)
        upper = triu(mat, 1)
        # Transpose lower and add to upper
        diag = diags(mat.diagonal())
        return diag + upper + lower.T
    
    counts_sp = symmetrize_sparse(counts_sp)
    counts_sp = csr_matrix(counts_sp, dtype=np.float32)
    
    #sparse_df = pd.DataFrame.sparse.from_spmatrix(counts_sp)
    
    rowsums = np.array(rowsums)
    rowsums = rowsums.flatten()
    
    colsums = np.array(colsums)
    colsums = colsums.flatten()

    sim_df = pd.DataFrame()
    sim_df['row_sums'] = rowsums
    sim_df['col_sums'] = colsums

    
    simulator = BaseSimulation(50, 50, max_dispersion_radius=50, max_dispersion_scale=50, joint_sums=sim_df)
    counts = simulator.simulate_experiment()#rowsums)
    coords = simulator.add_coords(simulator.bead_df)
    
    '''
    simulator = BaseSimulation(50, 50, max_dispersion_radius=50, max_dispersion_scale=50)
    counts = simulator.simulate_experiment(rowsums)
    coords = simulator.add_coords(simulator.bead_df)
    '''
    X_orig = coords[['x_coord','y_coord']].values
    true_dist = pairwise_distances(X_orig, metric='euclidean', n_jobs=-1)
    del X_orig
    counts = counts.pivot(index='source_bead', columns='target_bead', values='bead_counts')
    print(counts.shape)
    counts = counts.fillna(0.0)
    #counts[0] = 0
    counts = counts.reindex(index=coords.index, columns=coords.index, fill_value=0)
    print(counts.shape)
    #counts = counts.loc[:,sorted(counts.columns)]
    
    rowsums = counts.sum(axis=1)
    colsums = counts.sum(axis=0)
    sums = rowsums+colsums
    counts = counts/sums
    counts = (counts + counts.T)/2
    counts = counts.astype(np.float32)

    '''
    counts = counts.reset_index()
    print(counts.head())
    counts_long = pd.melt(counts, id_vars=["index"])
    counts_long.fillna(0.0, inplace=True)
    counts_long = counts_long[counts_long["value"]!=0.0]
    counts_long.iloc[:,0] = "bead_"+counts_long.iloc[:,0].astype(str)
    counts_long.iloc[:,1] = "bead_"+counts_long.iloc[:,1].astype(str)
    old_dir = dir.removesuffix("/figures/")
    print(old_dir)
    counts_long.to_csv(old_dir+"/sim.txt", sep="\t", header=False, index=False)


    def submit_job(script_path):
        # Submit the job and capture the job ID
        submit_command = ['qsub', script_path]
        result = subprocess.run(submit_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error submitting job: {result.stderr}")
            return None
        # Extract job ID from the output
        job_id = result.stdout.strip().split('.')[0]
        return job_id
        
    def check_job_status(job_id):
        # Check the job status
        status_command = ['qstat', job_id]
        result = subprocess.run(status_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # If qstat returns a non-zero code, it means the job is no longer in the queue
        return result.returncode == 0
        
    def wait_for_job_completion(job_id, check_interval=30):
        while check_job_status(job_id):
            print(f"Job {job_id} is still running. Checking again in {check_interval} seconds.")
            time.sleep(check_interval)
        print(f"Job {job_id} has completed.")
        return None

    qsub_path = "/net/gs/vol1/home/sanjayk/srivatsan/pipeline/sim_filter_edges.sh"

    job_id = submit_job(qsub_path)
    if job_id:
        print(f"Submitted job with ID {job_id}. Waiting for completion.")
        wait_for_job_completion(job_id)
        print("Job completed. Proceeding with the rest of the script.")
    else:
        print("Failed to submit job. Exiting.")

    
    counts = create_sparse_matrix_from_file(
        "/net/gs/vol1/home/sanjayk/srivatsan/pipeline/Asymmetric_filter_sim/simulation_filter/minpath_filtered_pairs_q0.2.txt")
    counts = pd.DataFrame(counts.toarray())
    '''
    print(counts.shape, flush=True)
    print(true_dist.shape, flush=True)
    
    dist_flatten = true_dist[np.triu_indices(true_dist.shape[0], k=1)]
    dist_flatten = dist_flatten.astype(np.float32)

    del true_dist
    avg_x = counts.values[np.triu_indices(counts.shape[0], k=1)]
    avg_x = 101*avg_x/(100*avg_x+1)
    avg_x = avg_x.astype(np.float32)

    indices = np.logical_not(np.logical_or(np.isnan(avg_x), np.isnan(dist_flatten)))
    indices = np.array(indices)
    print(len(avg_x),len(indices))
    avg_x = avg_x[indices]
    dist_flatten = dist_flatten[indices]
    
    means = binned_statistic(avg_x, dist_flatten, 
                             statistic='mean', 
                             bins=100, 
                             range=(0, 1.0))
    means_y = np.nan_to_num(means[0], nan=1).astype(np.float32)
    means_x = means[1]
    bins_mid = np.array([means_x[i] for i in range(len(means_x)-1)], dtype=np.float32)
    gbr = RandomForestRegressor(monotonic_cst=[-1], 
                            max_depth=10, 
                            criterion="friedman_mse",
                           n_estimators=100)
    gbr.fit(avg_x.reshape(-1,1), dist_flatten)
    
    x_pred = np.arange(0,1.0,0.001)
    y_pred = gbr.predict(x_pred.reshape(-1, 1))
    plt.scatter(avg_x, dist_flatten, s=0.5, label="Normalized Counts")
    plt.scatter(x_pred, y_pred, label="Predicted", s=0.5)
    plt.scatter(bins_mid, means_y, label="Moving average", s=0.5)
    plt.ylim([0,55])
    plt.legend()
    plt.xlabel("Normalized counts")
    plt.ylabel("Euclidean distance")
    plt.tight_layout()
    plt.savefig(dir+"/distance_curve.png", dpi=500)
    plt.show()

    return counts_sp, gbr


#@profile
def cluster_beads(counts_sp, gbr, max_size=2500, cluster_thresh=300, threshold=0.3):
    #G = igraph.Graph.Weighted_Adjacency(counts_sp)
    print("Creating graph...")
    time1 = timeit.default_timer()
    rows, cols = counts_sp.nonzero()
    #weights = counts_sp.data
    # Create an edge list
    edges = list(zip(rows, cols))
    G = ig.Graph(n=counts_sp.shape[0], directed=False)
    G.add_edges(edges)
    G.es["weight"] = counts_sp.data #weights

    del rows, cols, edges#, weights
    
    bead_idx = np.array(list(range(counts_sp.shape[0])))

    time2 = timeit.default_timer()
    print("Time to create graph: {}\n".format(datetime.timedelta(seconds=int(time2-time1))), flush=True)
    
    
    #partition2 = la.find_partition(G, la.ModularityVertexPartition, weights="weight", max_comm_size=2500)
    #clusters2 = partition2._membership

    print("Number of beads: {}".format(counts_sp.shape[0]), flush=True)

    print("Computing clustering...", flush=True)
    time1 = timeit.default_timer()

    '''
    def cluster_leiden(graph):
        partition_res = la.find_partition(graph, la.ModularityVertexPartition, weights="weight", max_comm_size=2500)
        return partition_res._membership
    results = Parallel(n_jobs=-1)(delayed(cluster_leiden)(G) for i in range(12))
    results = list(results)

    def consensus_clustering(cluster_runs, n_samples, max_cluster_size=2500):
        cluster_runs = np.array(cluster_runs)
        n_runs = cluster_runs.shape[0]
        
        co_association = np.zeros((n_samples, n_samples))
        
        for run in cluster_runs:
            unique_labels = np.unique(run)
            for label in unique_labels:
                cluster_mask = (run == label)
                co_association += np.outer(cluster_mask, cluster_mask)
        
        co_association /= n_runs
        
        # Convert to distance matrix
        distance_matrix = 1 - co_association
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distance_matrix, method='average')
        
        # Find optimal number of clusters
        for t in np.arange(0.1, 1.0, 0.1):
            labels = fcluster(linkage_matrix, t, criterion='distance')
            cluster_sizes = np.bincount(labels)
            if np.max(cluster_sizes) <= max_cluster_size:
                break
        return labels

    clusters2 = consensus_clustering(results, counts_sp.shape[0], 2500)
    '''

    partition2 = la.find_partition(G, la.ModularityVertexPartition, weights="weight", max_comm_size=max_size)
    clusters2 = partition2._membership

    del partition2
    
    time2 = timeit.default_timer()
    print("Time to compute clustering: {}\n".format(datetime.timedelta(seconds=int(time2-time1))), flush=True)
    
    
    #n_cluster = int(counts_sp.shape[0]/2500)
    #SC = SpectralClustering(n_clusters=n_cluster, affinity="precomputed")
    #clusters2 = SC.fit_predict(counts_sp)

    #####
    
    counter = dict(Counter(clusters2))

    # loop
    loop_bool = True
    count = 1
    overlap_thresh = 30
    #cluster_thresh = 200
    
    while loop_bool:
        counter = dict(Counter(clusters2))
        counter = pd.Series(counter)
        counter.sort_values(inplace=True)
        print(count, flush=True)
        ################
        #from collections import defaultdict
    
        # Initialize a defaultdict to store boundary nodes with keys as the top bordering cluster
        boundary_nodes = defaultdict(list)
        
        # Threshold for the ratio of edges connecting to nodes in different clusters
        #threshold = 0.3
        
        # Iterate through each node in the graph
        for node_index, node_partition in enumerate(clusters2):
            # Get the indices of nonzero elements in the row corresponding to the current node
            row_start = counts_sp.indptr[node_index]
            row_end = counts_sp.indptr[node_index + 1]
            row_nonzero_indices = counts_sp.indices[row_start:row_end]
            
            # Get the corresponding edge weights
            edge_weights = counts_sp.data[row_start:row_end]
            
            # Get the cluster assignments of neighboring nodes
            neighbor_clusters = [clusters2[neighbor_index] for neighbor_index in row_nonzero_indices]
            
            # Count the number of edges connecting to nodes in different clusters
            different_cluster_edges = sum(weight for neighbor_cluster, weight in zip(neighbor_clusters, edge_weights)
                                          if neighbor_cluster != node_partition)
            
            # Calculate the ratio of edges connecting to nodes in different clusters
            total_edges = sum(edge_weights)
            if total_edges > 0:
                ratio = different_cluster_edges / total_edges
            else:
                ratio = 0.0
            
            # Check if the ratio exceeds the threshold
            if ratio >= threshold:
                # Find the neighboring clusters with ratio above the threshold
                neighboring_clusters = set(neighbor_clusters)
                neighboring_clusters.discard(node_partition)  # Remove the node's cluster
                # Sort neighboring clusters by ratio in descending order
                sorted_neighbors = sorted(neighboring_clusters, 
                                          key=lambda cluster: sum(weight for neighbor_cluster, 
                                                                  weight in zip(neighbor_clusters, 
                                                                                edge_weights) if neighbor_cluster == cluster), reverse=True)
                # Find the top bordering cluster that is different from the node's cluster
                top_bordering_cluster = next((cluster for cluster in sorted_neighbors if cluster != node_partition), None)
                if top_bordering_cluster is not None:
                    # Add the node to the boundary nodes dictionary with the top bordering cluster as the key
                    boundary_nodes[top_bordering_cluster].append(node_index)
    
        ###############
        unique_clusters = np.unique(clusters2)
        cluster_dict = {}
        for i in unique_clusters:
            cluster_dict[i] = list(np.where(np.array(clusters2)==i)[0])
        total_dict = {}
        for i in cluster_dict.keys():
            total_dict[i] = np.unique(cluster_dict[i] + boundary_nodes[i])
        overlapping = {}
        for i in total_dict.keys():
            for j in total_dict.keys():
                if i != j:
                    inter = set(total_dict[i]).intersection(set(total_dict[j]))
                    if len(inter) > 0:
                        overlapping[(i,j)] = list(inter)
        for i in total_dict.keys():
            neighbors = [j for j in overlapping.keys() if i in j]
            for k in neighbors:
                total_dict[i] = np.unique(list(total_dict[i]) + list(overlapping[k]))
        overlap_sizes = pd.DataFrame(index=unique_clusters, columns=unique_clusters)
        overlap_sizes.fillna(0, inplace=True)
        for i in overlap_sizes.index:
            for j in overlap_sizes.index:
                try:
                    overlap_sizes.loc[(i,j)] = len(overlapping[(i,j)])
                except:
                    pass
        max_overlap = overlap_sizes.apply(np.max)
        empty_clusters = max_overlap[max_overlap < overlap_thresh].index
        empty_clusters = np.array(empty_clusters)
        print(empty_clusters, flush=True)
        for i in boundary_nodes.keys():
            print(i,len(boundary_nodes[i]), flush=True)
        ###############
        clusters2 = np.array(clusters2)
        mask = ~np.isin(clusters2, empty_clusters)
        indices = np.nonzero(mask)[0]
        clusters2 = clusters2[indices]
        counts_sp = counts_sp[indices, :].copy()
        counts_sp = counts_sp.tocsc()
        counts_sp = counts_sp[:, indices].copy()
        counts_sp = counts_sp.tocsr()
        #sparse_df = pd.DataFrame.sparse.from_spmatrix(counts_sp)
        bead_idx = bead_idx[indices]

        overlapping = {k: v for k, v in overlapping.items() 
               if k[0] not in empty_clusters and k[1] not in empty_clusters} #new added
        
        ##################
        clusters2 = np.array(clusters2)
        counter = dict(Counter(clusters2))
        counter = pd.Series(counter)
        counter.sort_values(inplace=True)
        counter = counter[counter<cluster_thresh]
        ################
    
        # Combining clusters
        while len(counter) > 0:
            for i in counter.index:
                if i in counter.index:
                    print(i,counter[i])
                    keys = [j for j in overlapping.keys() if j[0]==i or j[1]==i]
                    #overlaps = {l:len(overlapping[l]) for l in keys}
            
                    # merge with smallest neighboring cluster
                    counter2 = dict(Counter(clusters2))
                    counter2 = pd.Series(counter2)
                    
                    #not_yet = list(set(flatten(keys))-set([i]))
                    #list_sizes = [counter2[t] for t in not_yet]

                    not_yet = list(set(flatten(keys))-set([i]))
                    not_yet = [t for t in not_yet if t in counter2.index]
                    list_sizes = [counter2[t] for t in not_yet] #new added
                    
                    print(not_yet)
                    print(list_sizes)
                    if len(not_yet) > 0:
                        min_key = not_yet[np.argmin(list_sizes)]
                        print(min_key, min(list_sizes))
                        
                        # merging clusters
                        clusters2[clusters2==i] = min_key
                        unique_clusters = np.unique(clusters2)
                        cluster_dict = {}
                        for a in unique_clusters:
                            cluster_dict[a] = list(np.where(clusters2==a)[0])
                
                        boundary_nodes = defaultdict(list)
                
                        #threshold = 0.4
                        for node_index, node_partition in enumerate(clusters2):
                            row_start = counts_sp.indptr[node_index]
                            row_end = counts_sp.indptr[node_index + 1]
                            row_nonzero_indices = counts_sp.indices[row_start:row_end]
                            edge_weights = counts_sp.data[row_start:row_end]
                            neighbor_clusters = [clusters2[neighbor_index] for neighbor_index in row_nonzero_indices]
                            different_cluster_edges = sum(weight for neighbor_cluster, weight in zip(neighbor_clusters, edge_weights)
                                                          if neighbor_cluster != node_partition)
                            total_edges = sum(edge_weights)
                            if total_edges > 0:
                                ratio = different_cluster_edges / total_edges
                            else:
                                ratio = 0.0
                            if ratio >= threshold:
                                neighboring_clusters = set(neighbor_clusters)
                                neighboring_clusters.discard(node_partition)  # Remove the node's cluster
                                sorted_neighbors = sorted(neighboring_clusters, 
                                                          key=lambda cluster: sum(weight for neighbor_cluster, 
                                                                                  weight in zip(neighbor_clusters, 
                                                                                                edge_weights) 
                                                                                  if neighbor_cluster == cluster), reverse=True)
                                top_bordering_cluster = next((cluster for cluster in sorted_neighbors if cluster != node_partition), None)
                                if top_bordering_cluster is not None:
                                    boundary_nodes[top_bordering_cluster].append(node_index)
                        
                        total_dict = {}
                        for i in cluster_dict.keys():
                            total_dict[i] = cluster_dict[i] + boundary_nodes[i]
                        overlapping = {}
                        for b in total_dict.keys():
                            for c in total_dict.keys():
                                if b != c:
                                    inter = set(total_dict[b]).intersection(set(total_dict[c]))
                                    if len(inter) > 0:
                                        overlapping[(b,c)] = list(inter)
                        counter = dict(Counter(clusters2))
                        counter = pd.Series(counter)
                        counter.sort_values(inplace=True)
                        counter = counter[counter<cluster_thresh]
                    print("---------------------")
        ###################
        counter = dict(Counter(clusters2))
        counter = pd.Series(counter)
        counter.sort_values(inplace=True)
        clusters2 = np.array(clusters2)
        mask = ~np.isin(clusters2, empty_clusters)
        indices = np.nonzero(mask)[0]
        clusters2 = clusters2[indices]
        counts_sp = counts_sp[indices, :].copy()
        counts_sp = counts_sp.tocsc()
        counts_sp = counts_sp[:, indices].copy()
        counts_sp = counts_sp.tocsr()
        #sparse_df = pd.DataFrame.sparse.from_spmatrix(counts_sp)
        bead_idx = bead_idx[indices]
        ###################
        #from collections import defaultdict
    
        # Initialize a defaultdict to store boundary nodes with keys as the top bordering cluster
        boundary_nodes = defaultdict(list)
        
        # Threshold for the ratio of edges connecting to nodes in different clusters
        #threshold = 0.3
        
        # Iterate through each node in the graph
        for node_index, node_partition in enumerate(clusters2):
            # Get the indices of nonzero elements in the row corresponding to the current node
            row_start = counts_sp.indptr[node_index]
            row_end = counts_sp.indptr[node_index + 1]
            row_nonzero_indices = counts_sp.indices[row_start:row_end]
            
            # Get the corresponding edge weights
            edge_weights = counts_sp.data[row_start:row_end]
            
            # Get the cluster assignments of neighboring nodes
            neighbor_clusters = [clusters2[neighbor_index] for neighbor_index in row_nonzero_indices]
            
            # Count the number of edges connecting to nodes in different clusters
            different_cluster_edges = sum(weight for neighbor_cluster, weight in zip(neighbor_clusters, edge_weights)
                                          if neighbor_cluster != node_partition)
            
            # Calculate the ratio of edges connecting to nodes in different clusters
            total_edges = sum(edge_weights)
            if total_edges > 0:
                ratio = different_cluster_edges / total_edges
            else:
                ratio = 0.0
            
            # Check if the ratio exceeds the threshold
            if ratio >= threshold:
                # Find the neighboring clusters with ratio above the threshold
                neighboring_clusters = set(neighbor_clusters)
                neighboring_clusters.discard(node_partition)  # Remove the node's cluster
                # Sort neighboring clusters by ratio in descending order
                sorted_neighbors = sorted(neighboring_clusters, 
                                          key=lambda cluster: sum(weight for neighbor_cluster, 
                                                                  weight in zip(neighbor_clusters, 
                                                                                edge_weights) if neighbor_cluster == cluster), reverse=True)
                # Find the top bordering cluster that is different from the node's cluster
                top_bordering_cluster = next((cluster for cluster in sorted_neighbors if cluster != node_partition), None)
                if top_bordering_cluster is not None:
                    # Add the node to the boundary nodes dictionary with the top bordering cluster as the key
                    boundary_nodes[top_bordering_cluster].append(node_index)
    
        ###############
        unique_clusters = np.unique(clusters2)
        cluster_dict = {}
        for i in unique_clusters:
            cluster_dict[i] = list(np.where(np.array(clusters2)==i)[0])
        total_dict = {}
        for i in cluster_dict.keys():
            total_dict[i] = np.unique(cluster_dict[i] + boundary_nodes[i])
        overlapping = {}
        for i in total_dict.keys():
            for j in total_dict.keys():
                if i != j:
                    inter = set(total_dict[i]).intersection(set(total_dict[j]))
                    if len(inter) > 0:
                        overlapping[(i,j)] = list(inter)
        for i in total_dict.keys():
            neighbors = [j for j in overlapping.keys() if i in j]
            for k in neighbors:
                total_dict[i] = np.unique(list(total_dict[i]) + list(overlapping[k]))
        overlap_sizes = pd.DataFrame(index=unique_clusters, columns=unique_clusters)
        overlap_sizes.fillna(0, inplace=True)
        for i in overlap_sizes.index:
            for j in overlap_sizes.index:
                try:
                    overlap_sizes.loc[(i,j)] = len(overlapping[(i,j)])
                except:
                    pass
        max_overlap = overlap_sizes.apply(np.max)
        empty_clusters = max_overlap[max_overlap < overlap_thresh].index
        empty_clusters = np.array(empty_clusters)
        print(empty_clusters)
        if len(empty_clusters)>0:
            loop_bool=True
        else:
            loop_bool=False
        for i in boundary_nodes.keys():
            print(i,len(boundary_nodes[i]))
        ###############
        clusters2 = np.array(clusters2)
        mask = ~np.isin(clusters2, empty_clusters)
        indices = np.nonzero(mask)[0]
        clusters2 = clusters2[indices]
        counts_sp = counts_sp[indices, :].copy()
        counts_sp = counts_sp.tocsc()
        counts_sp = counts_sp[:, indices].copy()
        counts_sp = counts_sp.tocsr()
        #sparse_df = pd.DataFrame.sparse.from_spmatrix(counts_sp)
        bead_idx = bead_idx[indices]
        ##################
        count += 1

    print("Number of clusters: {}".format(len(np.unique(clusters2))))

    # end loop
    #return counts_sp, clusters2, bead_idx, cluster_dict, total_dict, overlapping, boundary_nodes#, sparse_df
    return counts_sp, clusters2, bead_idx


#@profile
def initialize_positions_fast(n_beads, cluster_assignments, paga_positions):
    """Vectorized position initialization."""
    positions = np.zeros((n_beads, 2), dtype=np.float32)
    unique_clusters = np.unique(cluster_assignments)
    
    # Vectorized initialization for all clusters
    noise_scale = 0.1
    noise = np.random.normal(0, noise_scale, size=(n_beads, 2)).astype(np.float32)
    
    for cluster in unique_clusters:
        cluster_mask = cluster_assignments == cluster
        positions[cluster_mask] = paga_positions[cluster]
    
    positions += noise
    return positions


#@profile
def initial_paga(counts_sp, gbr, clusters2, dir):
    print(counts_sp.shape)
    counts_norm = counts_sp.copy()
    print("Creating distance matrix...")
    time1 = timeit.default_timer()
    counts_norm[counts_norm.nonzero()] = 101*counts_norm[counts_norm.nonzero()]/(100*counts_norm[counts_norm.nonzero()]+1)
    counts_norm = counts_norm.astype(np.float32)
    dist_mat = counts_norm.copy()
    dist_mat[dist_mat.nonzero()] = gbr.predict(np.array(counts_norm[counts_norm.nonzero()].reshape(-1, 1))).reshape(1,-1).astype(np.float32)
    print(dist_mat.shape)
    dist_mat.setdiag(0)
    dist_mat.eliminate_zeros()
    time2 = timeit.default_timer()
    print("Time to create distance matrix: {}\n".format(datetime.timedelta(seconds=int(time2-time1))), flush=True)
    
    ####

    print("Computing initial PAGA...")
    adata = ad.AnnData(X=counts_norm)
    adata.X = adata.X.astype(np.float32)
    
    adata.obs['leiden_custom'] = clusters2
    adata.obs['leiden_custom'] = adata.obs['leiden_custom'].astype('category')
    adata.uns['neighbors'] = {
        'connectivities_key': 'connectivities',
        'distances_key': 'distances',
        'params': {
            'n_neighbors': adata.X.shape[0],  # or set an appropriate number
            'method': 'scanpy',
        }
    }
    adata.obsp['connectivities'] = adata.X
    adata.obsp['distances'] = dist_mat  # If your adjacency matrix also represents distances
    
    sc.tl.paga(adata, groups='leiden_custom')
    sc.pl.paga(adata, edge_width_scale=0.2, threshold=0.5)
    plt.tight_layout()
    plt.savefig(dir+"/initial_paga.png", dpi=500)
    plt.show()
    plt.gcf()
    plt.clf()

    uniq_clust = np.unique(clusters2)
    paga_positions = {uniq_clust[i]:adata.uns['paga']['pos'][i,:] for i in range(len(uniq_clust))}


    init_positions = initialize_positions_fast(counts_norm.shape[0], clusters2, paga_positions)

    min_x = min(init_positions[:,0])
    max_x = max(init_positions[:,0])
    range_x = max_x-min_x
    min_x = min_x-0.1*range_x
    max_x = max_x+0.1*range_x
    range_x = max_x-min_x
    min_y = min(init_positions[:,1])
    max_y = max(init_positions[:,1])
    range_y = max_y-min_y
    min_y = min_y-0.1*range_y
    max_y = max_y+0.1*range_y
    range_y = max_y-min_y
    centroid = np.mean(init_positions, axis=0)
    if range_x>range_y:
        min_y = centroid[1]-0.5*range_x
        max_y = centroid[1]+0.5*range_x
    else:
        min_x = centroid[0]-0.5*range_y
        max_x = centroid[0]+0.5*range_y
    
    plt.gcf()
    plt.clf()
    N = len(np.unique(clusters2))
    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    plt.scatter(init_positions[:,0], init_positions[:,1], s=2, cmap=cmap, edgecolors='black', linewidth=0.1, c=clusters2)#c=clusters2
    plt.colorbar()
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Initial PAGA")
    plt.tight_layout()
    plt.savefig(dir+"/initial_paga_pos.png", dpi=500)
    plt.show()
    plt.gcf()
    plt.clf()
    
    return dist_mat, counts_norm
    

#@profile
def process_dist(dist_mat, gbr, clusters2, bead_idx, counts_sp, dir):
    
    #print("Creating distance graph...")
    #G = nx.from_scipy_sparse_array(dist_mat)
    '''
    rows, cols = dist_mat.nonzero()
    weights = dist_mat.data
    G = nx.Graph()
    G.add_nodes_from(range(dist_mat.shape[0]))
    edges = [(rows[i], cols[i], {'weight': weights[i]}) for i in range(len(weights))]
    G.add_edges_from(edges)
    '''

    print("Removing low degree nodes...")
    time1 = timeit.default_timer()

    '''
    K=100
    degrees = dist_mat.getnnz(axis=1)
    indices = np.where(degrees>=K)[0]
    dist_100 = dist_mat.copy()
    dist_100 = dist_100[indices,:][:,indices].copy()
    counts_100 = counts_sp.copy()
    counts_100 = counts_100[indices,:][:,indices].copy()
    clusters_100 = clusters2.copy()
    clusters_100 = clusters_100[indices].copy()
    bead_idx = bead_idx[indices]
    '''

    ### snellen part
    
    rows, cols = dist_mat.nonzero()
    weights = dist_mat.data
    G = nx.Graph()
    G.add_nodes_from(range(dist_mat.shape[0]))
    edges = [(rows[i], cols[i], {'weight': weights[i]}) for i in range(len(weights))]
    G.add_edges_from(edges)
    
    #G = nx.from_scipy_sparse_array(dist_mat)

    ########
    degree_dist = [G.degree(n) for n in G.nodes()]
    plt.hist(degree_dist, bins=50)
    plt.tight_layout()
    plt.savefig(dir+"/initial_degree_dist.png", dpi=500)
    plt.show()
    plt.gcf()
    plt.clf()
    ########

    K=20 #100
    G_100 = nx.k_core(G, K)
    print("G size...")
    print(len(G.nodes()))
    print("G_100 size...")
    print(len(G_100.nodes()))
    
    dist_100 = nx.adjacency_matrix(G_100).astype(np.float32)
    degrees = dist_100.getnnz(axis=1)
    print(min(degrees))
    indices = np.array(G_100.nodes())

    #counts_100 = counts_norm.copy()
    counts_100 = counts_sp.copy()
    counts_100 = counts_100[indices,:][:,indices].copy()
    
    clusters_100 = clusters2.copy()
    clusters_100 = clusters_100[indices].copy()

    bead_idx = bead_idx[indices]
    
    ####

    
    time2 = timeit.default_timer()
    print("Time to remove low degree nodes: {}\n".format(datetime.timedelta(seconds=int(time2-time1))), flush=True)
    ####

    print("Filling in zeros...")
    time1 = timeit.default_timer()
    NN=250
    dist_mod = dist_100.copy()
    degrees = dist_mod.getnnz(axis=1)
    indices = np.where(degrees<=NN+1)[0]
    mod = dist_mod[indices,:].copy()
    idx_clusters = clusters_100[indices]
    max_dist = gbr.predict(np.array([0]).reshape(1,-1))[0].astype(np.float32)
    mod = csr_matrix(mod)

    for i in np.unique(idx_clusters):
        print(f"Processing cluster: {i}")
        # Get the row and column indices directly
        row_idx = np.where(idx_clusters == i)[0]
        col_idx = np.where(clusters_100 == i)[0]
        print(len(row_idx),len(col_idx))
        
        # Extract the submatrix
        mod_clust = mod[row_idx, :][:, col_idx].toarray()
        
        # Modify values
        mod_clust[mod_clust == 0] = max_dist
        
        # Update the original matrix using coordinate format
        mod = mod.tolil()  # Convert to LIL format for efficient matrix updates
        mod[row_idx[:, None], col_idx] = mod_clust    

    ''' # Crucial previous method
    #dist_mod = dist_mod.tolil()
    #dist_mod[indices,:] = mod
    '''

    
    dist_mod = dist_mod.tocsr().astype(np.float32)

    # new way of updating
    mod = mod.tocoo()

    # Modify row indices in-place to map from local to global
    mod.row = indices[mod.row]
    
    # Use the coordinates and values directly from the modified mod
    rows = mod.row       # Now contains global row indices
    cols = mod.col       # Original column indices 
    data = mod.data      # Values to update
    
    # Convert dist_mod to LIL format for efficient element updates
    dist_mod = dist_mod.tolil()
    
    # Directly update the elements at those coordinates
    for i in range(len(rows)):
        dist_mod[rows[i], cols[i]] = data[i]
    
    # Convert back to CSR for further operations
    dist_mod = dist_mod.tocsr()
    
    # Clean up
    dist_mod.setdiag(0)
    dist_mod.eliminate_zeros()

    
    time2 = timeit.default_timer()
    print("Time to fill in zeros: {}\n".format(datetime.timedelta(seconds=int(time2-time1))), flush=True)
    

    #return G, G_100, dist_100, dist_mod, counts_100, clusters_100, bead_idx
    return dist_100, dist_mod, counts_100, clusters_100, bead_idx


#@profile
def knn_paga(dist_100, dist_mod, counts_100, clusters_100, bead_idx, dir):
    print("Computing nearest neighbors...")
    time1 = timeit.default_timer()
    NN=250
    KNT = KNeighborsTransformer(mode="distance", metric="precomputed", n_jobs=-1, n_neighbors=NN)
    KNT.fit(dist_mod)
    neigh_dist, neigh_ind = KNT.kneighbors(dist_mod, return_distance=True)
    self_dist = np.zeros((dist_mod.shape[0], 1))
    neigh_dist = np.hstack([self_dist, neigh_dist[:, :-1]])
    neigh_ind = np.hstack([np.arange(dist_mod.shape[0]).reshape(-1, 1), neigh_ind[:, :-1]])

    dist_knt = KNT.transform(dist_mod).astype(np.float32)

    #####
    counts_100 = counts_100.tocoo()
    mask = np.isin(counts_100.col, dist_knt.indices)
    filtered_rows = counts_100.row[mask]
    filtered_cols = counts_100.col[mask]
    filtered_data = counts_100.data[mask]

    neigh_connect = csr_matrix((filtered_data, (filtered_rows, filtered_cols)), 
                          shape=counts_100.shape)
    '''
    neigh_connect = csr_matrix((counts_100.data[np.where(np.isin(counts_100.indices, dist_knt.indices))], 
                       counts_100.indices[np.where(np.isin(counts_100.indices, dist_knt.indices))], 
                       counts_100.indptr), 
                      shape=counts_100.shape)
    '''
    #####
    
    uniq_clust = np.unique(clusters_100)

    time2 = timeit.default_timer()
    print("Time to compute NN: {}\n".format(datetime.timedelta(seconds=int(time2-time1))), flush=True)
    

    print("Computing final PAGA...")
    time1 = timeit.default_timer()
    adata = ad.AnnData(X=neigh_connect)
    adata.X = adata.X.astype(np.float32)
    
    adata.obs['leiden_custom'] = clusters_100
    adata.obs['leiden_custom'] = adata.obs['leiden_custom'].astype('category')
    adata.uns['neighbors'] = {
        'connectivities_key': 'connectivities',
        'distances_key': 'distances',
        'params': {
            'n_neighbors': adata.X.shape[0],  # or set an appropriate number
            'method': 'scanpy',
        }
    }
    adata.obsp['connectivities'] = adata.X
    adata.obsp['distances'] = dist_knt  # If your adjacency matrix also represents distances
    
    sc.tl.paga(adata, groups='leiden_custom')
    sc.pl.paga(adata, edge_width_scale=0.2, threshold=0.5)#, pos=pos_new)
    plt.tight_layout()
    plt.savefig(dir+"/final_paga.png", dpi=500)
    plt.show()

    time2 = timeit.default_timer()
    print("Time to compute PAGA: {}\n".format(datetime.timedelta(seconds=int(time2-time1))), flush=True)

    
    paga_positions = {uniq_clust[i]:adata.uns['paga']['pos'][i,:] for i in range(len(uniq_clust))}
    init_positions = initialize_positions_fast(counts_100.shape[0], clusters_100, paga_positions)

    min_x = min(init_positions[:,0])
    max_x = max(init_positions[:,0])
    range_x = max_x-min_x
    min_x = min_x-0.1*range_x
    max_x = max_x+0.1*range_x
    range_x = max_x-min_x
    min_y = min(init_positions[:,1])
    max_y = max(init_positions[:,1])
    range_y = max_y-min_y
    min_y = min_y-0.1*range_y
    max_y = max_y+0.1*range_y
    range_y = max_y-min_y
    centroid = np.mean(init_positions, axis=0)
    if range_x>range_y:
        min_y = centroid[1]-0.5*range_x
        max_y = centroid[1]+0.5*range_x
    else:
        min_x = centroid[0]-0.5*range_y
        max_x = centroid[0]+0.5*range_y
    
    plt.gcf()
    plt.clf()
    N = len(np.unique(clusters_100))
    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    plt.scatter(init_positions[:,0], init_positions[:,1], s=2, cmap=cmap, edgecolors='black', linewidth=0.1, c=clusters_100)#c=clusters2
    plt.colorbar()
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Final PAGA")
    plt.tight_layout()
    plt.savefig(dir+"/final_paga_pos.png", dpi=500)
    plt.show()

    paga_df = pd.DataFrame(paga_positions)
    paga_df.to_csv(dir+"final_paga_pos.csv")

    ##############
    print("Computing Delaunay triangulation and spring layout...")
    time1 = timeit.default_timer()

    points = np.array([paga_positions[k] for k in sorted(paga_positions.keys())])
    points = 100*points
    tri = Delaunay(points)
    G = nx.Graph()
    for i, k in enumerate(sorted(paga_positions.keys())):
        G.add_node(k, pos=paga_positions[k])
    edge_list = set()
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i+1, len(simplex)):
                node_i = sorted(paga_positions.keys())[simplex[i]]
                node_j = sorted(paga_positions.keys())[simplex[j]]
                if (node_i, node_j) not in edge_list and (node_j, node_i) not in edge_list:
                    G.add_edge(node_i, node_j, weight=1.0)  # Equal weights
                    edge_list.add((node_i, node_j))

    pos_spring = nx.spring_layout(G, k=1, weight=None, iterations=200, threshold=1e-8)#, pos={k: paga_positions[k]*100 for k in G.nodes()}, fixed=None, 
                                  #k=0.1, iterations=50, weight='weight')

    spring_positions = {k: 100*pos_spring[k] for k in paga_positions.keys()}
    plt.gcf()
    plt.clf()

    nx.draw_networkx_edges(G, pos=spring_positions, width=1.0, alpha=0.5)
    
    # Draw nodes with same coloring as before
    node_colors = [clusters_100[0] for k in sorted(spring_positions.keys())]  # Use first point of each cluster for color
    nx.draw_networkx_nodes(G, pos=spring_positions, node_size=100, 
                          node_color=node_colors, cmap=cmap)
    
    #plt.xlim(min_x, max_x)
    #plt.ylim(min_y, max_y)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Spring Layout from Delaunay Triangulation")
    plt.tight_layout()
    plt.savefig(dir+"/spring_delaunay_layout.png", dpi=500)
    plt.show()
    
    time2 = timeit.default_timer()
    print("Time to compute Delaunay and spring layout: {}\n".format(datetime.timedelta(seconds=int(time2-time1))), flush=True)


    
    init_positions = initialize_positions_fast(counts_100.shape[0], clusters_100, spring_positions)
    min_x = min(init_positions[:,0])
    max_x = max(init_positions[:,0])
    range_x = max_x-min_x
    min_x = min_x-0.1*range_x
    max_x = max_x+0.1*range_x
    range_x = max_x-min_x
    min_y = min(init_positions[:,1])
    max_y = max(init_positions[:,1])
    range_y = max_y-min_y
    min_y = min_y-0.1*range_y
    max_y = max_y+0.1*range_y
    range_y = max_y-min_y
    centroid = np.mean(init_positions, axis=0)
    if range_x>range_y:
        min_y = centroid[1]-0.5*range_x
        max_y = centroid[1]+0.5*range_x
    else:
        min_x = centroid[0]-0.5*range_y
        max_x = centroid[0]+0.5*range_y
    
    plt.gcf()
    plt.clf()
    N = len(np.unique(clusters_100))
    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    plt.gcf()
    plt.clf()
    N = len(np.unique(clusters_100))
    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    plt.scatter(init_positions[:,0], init_positions[:,1], s=2, cmap=cmap, edgecolors='black', linewidth=0.1, c=clusters_100)#c=clusters2
    plt.colorbar()
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Final PAGA")
    plt.tight_layout()
    plt.savefig(dir+"/final_delaunay_pos.png", dpi=500)
    plt.show()

    return neigh_dist, neigh_ind, dist_knt, init_positions


#@profile
def umap_grid(neigh_dist, neigh_ind, dist_knt, init_positions, bead_idx, unique, clusters_100, dir, sample_dir, num_iter):
    print("Computing initial UMAP...")
    NN=250
    umap = UMAP(n_neighbors=NN, min_dist=0.1, n_epochs=5000, n_components=2, init=init_positions*100,
           precomputed_knn=(neigh_ind, neigh_dist.astype(np.float32)), verbose=True, local_connectivity=2, learning_rate=1.0, 
            low_memory=False, negative_sample_rate=2, repulsion_strength=1.0)

    umap_X = umap.fit_transform(dist_knt).astype(np.float32)

    min_x = min(umap_X[:,0])
    max_x = max(umap_X[:,0])
    range_x = max_x-min_x
    min_x = min_x-0.1*range_x
    max_x = max_x+0.1*range_x
    range_x = max_x-min_x
    min_y = min(umap_X[:,1])
    max_y = max(umap_X[:,1])
    range_y = max_y-min_y
    min_y = min_y-0.1*range_y
    max_y = max_y+0.1*range_y
    range_y = max_y-min_y
    centroid = np.mean(umap_X, axis=0)
    if range_x>range_y:
        min_y = centroid[1]-0.5*range_x
        max_y = centroid[1]+0.5*range_x
    else:
        min_x = centroid[0]-0.5*range_y
        max_x = centroid[0]+0.5*range_y
    
    plt.gcf()
    plt.clf()
    N = len(np.unique(clusters_100))
    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    plt.scatter(umap_X[:,0], umap_X[:,1], s=2, cmap=cmap, edgecolors='black', linewidth=0.1, c=clusters_100)#c=clusters2
    plt.colorbar()
    plt.xlim(min_x+1.0, max_x+1.0)
    plt.ylim(min_y+1.0, max_y+1.0)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Initial UMAP")
    plt.tight_layout()
    plt.savefig(dir+"/umap_initial.png", dpi=500)
    plt.show()
    
    ####
    umap_stitch = pd.DataFrame(umap_X, columns=['X0','X1'], dtype=np.float32)#, index=bead_idx)
    umap_stitch = umap_stitch.groupby(level=0).mean()

    umap_stitch['cluster'] = clusters_100
    umap_stitch.index = bead_idx
    umap_stitch.to_csv("{}initial_solution.csv".format(sample_dir))

    umap_barcodes = umap_stitch.copy()
    mapping_rev = dict(zip(list(range(len(unique))), unique))
    umap_barcodes.index = umap_barcodes.index.map(mapping_rev)
    umap_barcodes.to_csv("{}initial_solution_barcodes.csv".format(sample_dir))


    def run_umap(min_dist, repulsion_strength):
        umap = UMAP(n_neighbors=NN, min_dist=min_dist, n_epochs=num_iter, n_components=2, init=init_positions*100,
                precomputed_knn=(neigh_ind, neigh_dist.astype(np.float32)), local_connectivity=2, learning_rate=1.0, 
                low_memory=False, negative_sample_rate=2, repulsion_strength=repulsion_strength)
        umap_X = umap.fit_transform(dist_knt)
    
        KNT_umap = KNeighborsTransformer(mode="distance", n_jobs=-1, n_neighbors=NN)
        KNT_umap.fit(umap_X)
        neigh_ind_umap = KNT_umap.kneighbors(umap_X, return_distance=False)
        jaccard = np.zeros(neigh_ind.shape[0])
        for i in range(neigh_ind.shape[0]):
            #print(i)
            set1 = set(neigh_ind[i,:])
            set2 = set(neigh_ind_umap[i,:])
            inter = len(set1.intersection(set2))
            union = len(set1.union(set2))
            jaccard[i] = inter/union
        
        return umap_X, min_dist, repulsion_strength, jaccard
    md_range = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5]
    rep_strengths = [1.0,5.0]


    print("Computing UMAP gridsearch...")
    time1 = timeit.default_timer()
    results = Parallel(n_jobs=-1, max_nbytes=None)(delayed(run_umap)(i,j) for i in md_range for j in rep_strengths)
    results = list(results)
    #results = np.array(results)
    time2 = timeit.default_timer()
    print("Time to compute gridsearch: {}\n".format(datetime.timedelta(seconds=int(time2-time1))), flush=True)
    #print("gridsearch done.")

    
    def point_cloud_to_binary_image(point_cloud):
        # Compute the average Delaunay triangulation distance
        #tri = Delaunay(point_cloud)
    
        def median_dist(mat):
            #mat = np.multiply(mat.values, vars)
            noise = 1e-5 * np.random.rand(*mat.shape)
            mat += noise
            tri = Delaunay(mat, qhull_options="Qbb Qc Qz Q12")
            triangles = mat[tri.simplices]
        
            edge_lengths = []
            for simplex in tri.simplices:
                # Get the vertices of the triangle
                v0, v1, v2 = mat[simplex]
                
                # Calculate the lengths of the edges
                length = np.array([
                    euclidean(v0, v1),
                    euclidean(v1, v2),
                    euclidean(v2, v0)])
                edge_lengths.append(length)
                
            edge_lengths = np.array(edge_lengths).flatten()
        
            # Reshape edge_lengths to match triangles
            #edge_lengths = edge_lengths.reshape(-1, 3)
            median = np.median(edge_lengths)
            return median
        
        avg_dist = median_dist(point_cloud)
    
        # Determine the bounding box of the point cloud
        min_x, min_y = np.min(point_cloud, axis=0)
        max_x, max_y = np.max(point_cloud, axis=0)
    
        grid_size = 3.0 * avg_dist
    
        # Create a grid with the desired size
        width = int((max_x - min_x) // grid_size + 1)
        height = int((max_y - min_y) // grid_size + 1)
        binary_image = np.zeros((height, width), dtype=np.uint8)
    
        # Build a BallTree for efficient point lookup
        ball_tree = BallTree(point_cloud)
    
        # Iterate over the grid cells
        for i in range(height):
            for j in range(width):
                x_min = min_x + j * grid_size
                x_max = x_min + grid_size
                y_min = min_y + i * grid_size
                y_max = y_min + grid_size
    
                # Check if any point is within the grid cell
                indices = ball_tree.query_radius([(x_min, y_min), (x_max, y_max)], grid_size * np.sqrt(2))[0]
                if indices.size > 0:
                    binary_image[i, j] = 1
    
        return binary_image

    def analyze_binary_image(binary_image):
        # Erode the binary image
        eroded_image = binary_erosion(binary_image, np.ones((3,3)))
        eroded_image = binary_erosion(eroded_image, np.ones((3,3)))
        eroded_image = binary_erosion(eroded_image, np.ones((3,3)))
        eroded_image = binary_erosion(eroded_image, np.ones((3,3)))
        eroded_image = binary_erosion(eroded_image, np.ones((3,3)))
        #eroded_image = binary_erosion(eroded_image, np.ones((3,3)))
    
        # Perform watershed segmentation
        #segmented_image = chan_vese(eroded_image)#distance, markers, mask=eroded_image)
        #eroded_image = binary_image.copy()
        segmented_image = chan_vese(eroded_image)
        
        # Analyze the segmented image
        labeled_image, num_objects = label(segmented_image, connectivity=1, return_num=True)
        object_sizes = [np.sum(labeled_image == label) for label in range(1, num_objects + 1)]
    
        return num_objects, object_sizes, eroded_image

    def square_tiled_alpha_shape_2d_histogram(points, n_bins_total=1000, alpha=1.0):
        """
        Compute a 2D histogram on a point cloud using alpha shapes,
        tiling the shape with a specified number of square bins.
        Also computes a chi-square test statistic comparing the distribution
        to a uniform distribution.
        
        Args:
        points (np.array): Nx2 array of point coordinates
        n_bins_total (int): Approximate total number of square bins to cover the alpha shape
        alpha (float): Alpha value for the alpha shape (lower for tighter fit)
        
        Returns:
        tuple: (histogram, bin_size, x_min, y_min, alpha_shape, chi_square_statistic, p_value)
        """
        # Compute alpha shape
        alpha_shape = alphashape(points+1e-5 * np.random.rand(*points.shape), alpha)
        
        # Calculate bin size
        alpha_shape_area = alpha_shape.area
        bin_size = np.sqrt(alpha_shape_area / n_bins_total)
        
        # Find the minimum x and y coordinates of points within the alpha shape
        points_in_shape = [p for p in points if alpha_shape.contains(Point(p))]
        x_min, y_min = np.min(points_in_shape, axis=0)
        
        # Initialize histogram
        hist = defaultdict(int)
        
        # Bin points
        for point in points_in_shape:
            x_idx = int((point[0] - x_min) / bin_size)
            y_idx = int((point[1] - y_min) / bin_size)
            hist[(x_idx, y_idx)] += 1
        
        # Prepare data for chi-square test
        observed_frequencies = list(hist.values())
        n_bins = len(observed_frequencies)
        expected_frequency = len(points_in_shape) / n_bins
        expected_frequencies = [expected_frequency] * n_bins
        
        # Perform chi-square test
        chi_square_statistic, p_value = chisquare(observed_frequencies, f_exp=expected_frequencies)
        
        return hist, bin_size, x_min, y_min, alpha_shape, chi_square_statistic, p_value

    #######
    mpl.rcParams.update({'font.size': 6})
    fig, axs = plt.subplots(6,3, figsize=(15,30))
    axs = axs.flatten()
    for k in range(len(results)):
        ax = axs[k]
        ax.scatter(results[k][0][:,0], results[k][0][:,1], s=1, 
            c=results[k][3], cmap="Spectral_r", edgecolors='black', linewidth=0.1)
        ax.set_title("min_dist={}, rep_str={}, avg_jaccard={}".format(results[k][1], results[k][2], np.mean(results[k][3])))
        #ax.xaxis.set_major_formatter(plt.NullFormatter())
        #ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.set_xticks([])
        ax.set_yticks([])
    
    
        ax.set_aspect('equal')
    #plt.tight_layout()
    plt.tight_layout()
    plt.savefig(dir+"umap_grid_final.png", dpi=500)
    plt.show()
    plt.gcf()
    plt.clf()


    ######
    obj_bool = []
    
    fig2, axs2 = plt.subplots(6,3, figsize=(15,30))
    axs2 = axs2.flatten()
    for j in range(len(results)):
        print("Detecting number of objects...")
        print(j)
        points = results[j][0]
        binary_img = point_cloud_to_binary_image(points)
        
        num_objects, object_sizes, binary_img = analyze_binary_image(binary_img)
        
        object_sizes = np.array(object_sizes)
        object_sizes = object_sizes[object_sizes>50]
        print(len(object_sizes))

        if len(object_sizes) == 1:
            obj_bool.append(True)
        else:
            obj_bool.append(False)
    
        ax2 = axs2[j]
        ax2.imshow(binary_img[::-1,:].copy(), cmap="Greys")
        ax2.set_title("min_dist {}, rep_str={}, num_obj={}".format(results[j][1], results[j][2], len(object_sizes)))
        
    plt.tight_layout()
    plt.savefig(dir+"/binary_img_final.png", dpi=500)
    plt.show()
    plt.gcf()
    plt.clf()

    
    obj_bool = np.array(obj_bool)  # Make sure this is a NumPy array
    new_results = [results[i] for i in range(len(results)) if obj_bool[i]]
    #del results
    print("segmented bool: {}".format(obj_bool))
    #new_results = results[obj_bool]
    perp_range = [(i,j) for i in md_range for j in rep_strengths]
    new_params = np.array(perp_range)[obj_bool]
    print(new_params)

    #######
    n = len(results)  # Number of subplots
    fig3, axs3= plt.subplots(6,3, figsize=(15,30))
    axs3 = axs3.flatten()
    
    #hist_list = []
    stdevs = []
    for k in range(len(results)):
        print(k)
        mat = results[k][0]
        bins = int(mat.shape[0]/100)
        hist, bin_size, x_min, y_min, alpha_shape, chi_square, p_value = square_tiled_alpha_shape_2d_histogram(mat, bins)
        #hist_list.append(hist)
        stdevs.append(chi_square)
        #ax = axs[k]
        #ax.imshow(H, origin='lower', cmap='viridis')
        #ax.set_title("Perp={}".format(new_params[k]))
    
        ax3 = axs3[k]
        for (x_idx, y_idx), count in hist.items():
            rect = plt.Rectangle((x_min + x_idx * bin_size, y_min + y_idx * bin_size), 
                                 bin_size, bin_size, 
                                 #facecolor='blue', alpha=((np.log10(count+0.01)-np.log10(0.01)) / (np.log10(max(hist.values())+0.01)-np.log10(0.01))))
                                 facecolor='blue', alpha=count/max(hist.values()))
            ax3.add_patch(rect)
        
        ax3.set_xlim(x_min, x_min + (max(x_idx for x_idx, _ in hist.keys()) + 1) * bin_size)
        ax3.set_ylim(y_min, y_min + (max(y_idx for _, y_idx in hist.keys()) + 1) * bin_size)
        ax3.set_title("min_dist {}, rep_str={}, chi_square={}".format(results[k][1], results[k][2], round(chi_square)))
        ax3.set_aspect('equal')
        
    plt.tight_layout()
    plt.savefig(dir+"/hist2d_final.png", dpi=500)
    plt.show()
    plt.gcf()
    plt.clf()

    ######
    print(new_params)
    #print(new_results.shape)
    #print(new_results[0].shape)

    #stdevs = np.array(stdevs)
    #umap_X = new_results[np.argmin(stdevs[obj_bool])][0]
    stdevs = np.array(stdevs)
    stdevs_filtered = stdevs[obj_bool]  # Get only the stdev values where obj_bool is True
    min_index = np.argmin(stdevs_filtered)  # Find the index of the minimum in the filtered array
    umap_X = new_results[min_index][0]
    print("Accepted parameters: {}".format(new_params[np.argmin(stdevs[obj_bool])]))


    ######
    min_x = min(umap_X[:,0])
    max_x = max(umap_X[:,0])
    range_x = max_x-min_x
    min_x = min_x-0.1*range_x
    max_x = max_x+0.1*range_x
    range_x = max_x-min_x
    min_y = min(umap_X[:,1])
    max_y = max(umap_X[:,1])
    range_y = max_y-min_y
    min_y = min_y-0.1*range_y
    max_y = max_y+0.1*range_y
    range_y = max_y-min_y
    centroid = np.mean(umap_X, axis=0)
    if range_x>range_y:
        min_y = centroid[1]-0.5*range_x
        max_y = centroid[1]+0.5*range_x
    else:
        min_x = centroid[0]-0.5*range_y
        max_x = centroid[0]+0.5*range_y
    
    plt.gcf()
    plt.clf()
    N = len(np.unique(clusters_100))
    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    plt.scatter(umap_X[:,0], umap_X[:,1], s=2, cmap=cmap, edgecolors='black', linewidth=0.1, c=clusters_100)#c=clusters2
    plt.colorbar()
    plt.xlim(min_x+1.0, max_x+1.0)
    plt.ylim(min_y+1.0, max_y+1.0)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Final UMAP")
    plt.tight_layout()
    plt.savefig(dir+"/umap_final.png", dpi=500)
    plt.show()
    #####

    umap_stitch = pd.DataFrame(umap_X, columns=['X0','X1'], dtype=np.float32)#, index=bead_idx)
    umap_stitch = umap_stitch.groupby(level=0).mean()

    umap_stitch['cluster'] = clusters_100
    umap_stitch.index = bead_idx
    umap_stitch.to_csv("{}final_solution.csv".format(sample_dir))

    umap_barcodes = umap_stitch.copy()
    mapping_rev = dict(zip(list(range(len(unique))), unique))
    umap_barcodes.index = umap_barcodes.index.map(mapping_rev)
    umap_barcodes.to_csv("{}final_solution_barcodes.csv".format(sample_dir))

    
    del stdevs, stdevs_filtered, new_results, obj_bool
    
    return umap_stitch, umap_barcodes

    
def main_real():
    # Setup memory tracking
    '''tracemalloc.start()'''
    mem_snapshots = []
    
    time_start = timeit.default_timer()

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument('-s', '--sample_id', help = "Sample ID", required = True, default = "")
    parser.add_argument('-i', '--interactions_path', help = "Path to interactions csv", required = True, default = "")
    parser.add_argument('-m', '--max_size', help = "Max cluster size", required = True, default = "")
    parser.add_argument('-n', '--num_iter', help = "Number of UMAP iterations", required = True, default = "")
    parser.add_argument('-p', '--percent_downsample', help = "Downsample percentage", 
                        required = False, default = "")
    
    
    argument = parser.parse_args()
    sample_id = str(argument.sample_id)
    int_path = str(argument.interactions_path)
    max_size = int(argument.max_size)
    num_iter = int(argument.num_iter)

    try:
        pct = float(argument.percent_downsample)
    except:
        pct = 1.0
    
    sample_dir = "./{}/".format(sample_id)
    os.mkdir(sample_dir)
    dir = "./{}/figures/".format(sample_id)
    os.mkdir(dir)

    '''
    # Get initial memory usage
    current_mem = memory_usage()[0] / 1024  # Convert to GB
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024 * 1024)  # Convert to GB
    mem_snapshots.append(("Initial", current_mem, peak_mem))
    print(f"Memory: {current_mem:.2f} GB (Peak: {peak_mem:.2f} GB)")
    '''
    
    # Step 1: Load and create sparse matrix
    counts_sp, unique = create_sparse_matrix_from_file(int_path)
    '''current_mem = memory_usage()[0] / 1024
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024 * 1024)
    mem_snapshots.append(("After loading data", current_mem, peak_mem))
    print(f"After loading data - Memory: {current_mem:.2f} GB (Peak: {peak_mem:.2f} GB)")'''
    print("Sparse matrix dimensions: {}".format(counts_sp.shape))


    try:
        pct = float(argument.percent_downsample)
        counts_sp = downsample(counts_sp, pct)
        print("After downsampling: {}".format(counts_sp.shape))
    except:
        pass
    

    # Step 2: Calculate similarity distance mapping
    counts_sp, gbr = similarity_distance_mapping(dir, counts_sp)
    '''current_mem = memory_usage()[0] / 1024
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024 * 1024)
    mem_snapshots.append(("After distance mapping", current_mem, peak_mem))
    print(f"After distance mapping - Memory: {current_mem:.2f} GB (Peak: {peak_mem:.2f} GB)")'''

    # Step 3: Cluster beads - with reduced return values
    counts_sp, clusters2, bead_idx = cluster_beads(counts_sp, gbr, max_size=max_size, cluster_thresh=1001)
    '''current_mem = memory_usage()[0] / 1024
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024 * 1024)
    mem_snapshots.append(("After clustering", current_mem, peak_mem))
    print(f"After clustering - Memory: {current_mem:.2f} GB (Peak: {peak_mem:.2f} GB)")'''

    # Step 4: Initial PAGA computation
    dist_mat, counts_norm = initial_paga(counts_sp, gbr, clusters2, dir)
    '''current_mem = memory_usage()[0] / 1024
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024 * 1024)
    mem_snapshots.append(("After initial PAGA", current_mem, peak_mem))
    print(f"After initial PAGA - Memory: {current_mem:.2f} GB (Peak: {peak_mem:.2f} GB)")'''
    
    # Free memory for counts_sp as it's no longer needed
    del counts_sp
    '''current_mem = memory_usage()[0] / 1024
    print(f"After deleting counts_sp - Memory: {current_mem:.2f} GB")'''

    # Step 5: Process distances - with reduced return values
    dist_100, dist_mod, counts_100, clusters_100, bead_idx = process_dist(dist_mat, gbr, 
                                                                          clusters2, bead_idx, counts_norm, dir)
    '''current_mem = memory_usage()[0] / 1024
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024 * 1024)
    mem_snapshots.append(("After processing distances", current_mem, peak_mem))
    print(f"After processing distances - Memory: {current_mem:.2f} GB (Peak: {peak_mem:.2f} GB)")'''
    
    # Free memory for variables no longer needed
    del dist_mat, gbr, clusters2, counts_norm
    '''current_mem = memory_usage()[0] / 1024
    print(f"After cleanup - Memory: {current_mem:.2f} GB")'''
    
    # Step 6: Compute KNN PAGA - with modified parameters
    neigh_dist, neigh_ind, dist_knt, init_positions = knn_paga(dist_100, dist_mod, counts_100, clusters_100, bead_idx, dir)
    '''current_mem = memory_usage()[0] / 1024
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024 * 1024)
    mem_snapshots.append(("After KNN PAGA", current_mem, peak_mem))
    print(f"After KNN PAGA - Memory: {current_mem:.2f} GB (Peak: {peak_mem:.2f} GB)")'''
    
    # Free memory for variables no longer needed
    del dist_100, dist_mod, counts_100
    '''current_mem = memory_usage()[0] / 1024
    print(f"After cleanup - Memory: {current_mem:.2f} GB")'''
    
    # Step 7: Run UMAP grid search
    print("Starting UMAP grid search...")
    '''current_mem = memory_usage()[0] / 1024
    print(f"Before UMAP grid - Memory: {current_mem:.2f} GB")'''
    
    umap_stitch, umap_barcodes = umap_grid(neigh_dist, neigh_ind, dist_knt, init_positions, bead_idx, unique, clusters_100, dir, sample_dir, num_iter)
    '''current_mem = memory_usage()[0] / 1024
    peak_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024 * 1024)
    mem_snapshots.append(("After UMAP grid", current_mem, peak_mem))
    print(f"After UMAP grid - Memory: {current_mem:.2f} GB (Peak: {peak_mem:.2f} GB)")'''
    
    # Free memory for variables no longer needed
    del neigh_dist, neigh_ind, dist_knt, init_positions, bead_idx, unique, clusters_100
    '''current_mem = memory_usage()[0] / 1024
    print(f"After final cleanup - Memory: {current_mem:.2f} GB")'''
    
    # Print summary report
    time_end = timeit.default_timer()
    print("\nTotal script time: {}".format(datetime.timedelta(seconds=int(time_end-time_start))))
    
    '''print("\nMemory Usage Summary (GB):")
    print("-------------------------")
    for i in range(len(mem_snapshots)):
        step, mem, peak = mem_snapshots[i]
        print(f"{step}: {mem:.2f} GB (Peak: {peak:.2f} GB)")
        if i > 0:
            prev_mem = mem_snapshots[i-1][1]
            diff = mem - prev_mem
            print(f"   Change: {diff:.2f} GB ({'↑' if diff > 0 else '↓'})")
    
    tracemalloc.stop()
    '''

if __name__=="__main__":
    main_real()



