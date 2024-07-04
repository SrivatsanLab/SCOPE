from sklearn.metrics import pairwise_distances
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from scipy.optimize import minimize
from scipy.stats import binned_statistic
import timeit
import datetime
import matplotlib as mpl
import os
import sys
from sklearn.manifold import SpectralEmbedding
from sklearn.ensemble import RandomForestRegressor
import joblib
from joblib import Parallel, delayed
import igraph
import leidenalg as la
from collections import Counter
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
from scipy.optimize import basinhopping

sys.path.append('/net/gs/vol1/home/sanjayk/srivatsan/sci-space-v2')
from ssv2.simulation import BaseSimulation

from scipy.sparse import csr_matrix

from scipy.spatial import KDTree
from alphashape import alphashape
from shapely.geometry import Point

from sklearn.neighbors import BallTree

import skimage
from skimage.morphology import binary_erosion
from skimage.segmentation import watershed, chan_vese
from skimage.measure import label, regionprops

import math
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

import argparse
import scanpy as sc
from scipy.optimize import linear_sum_assignment

mpl.rcParams['figure.dpi'] = 500
np.random.seed(0)



def create_sparse_matrix_from_file(file_path, r1='R1_full_bc', r2='R2_full_bc', count='count'):
    df = pd.read_csv(file_path)

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

    sp = csr_matrix((values, (row_indices, col_indices)), shape=(nrows, ncols))
    # Create and return the sparse matrix
    return sp, unique


def similarity_distance_mapping(dir, counts_sp):
    rowsums = np.array(counts_sp.sum(axis=1)).flatten()
    colsums = np.array(counts_sp.sum(axis=0)).flatten()
    sums = rowsums+colsums
    r,c = counts_sp.nonzero()
    rD_sp = csr_matrix(((1.0/sums)[r], (r,c)), shape=(counts_sp.shape))
    counts_sp = counts_sp.multiply(rD_sp)
    counts_sp = (counts_sp + counts_sp.T)/2
    counts_sp = csr_matrix(counts_sp)
    #sparse_df = pd.DataFrame.sparse.from_spmatrix(counts_sp)

    rowsums = np.array(rowsums)
    rowsums = rowsums.flatten()
    simulator = BaseSimulation(50, 50, max_dispersion_radius=50, max_dispersion_scale=50)
    counts = simulator.simulate_experiment(rowsums)
    coords = simulator.add_coords(simulator.bead_df)
    X_orig = coords[['x_coord','y_coord']].values
    true_dist = pairwise_distances(X_orig, metric='euclidean', n_jobs=-1)
    counts = counts.pivot(index='source_bead', columns='target_bead', values='bead_counts')
    counts = counts.fillna(0.0)
    counts[0] = 0
    counts = counts.loc[:,sorted(counts.columns)]
    rowsums = counts.sum(axis=1)
    colsums = counts.sum(axis=0)
    sums = rowsums+colsums
    counts = counts/sums
    counts = (counts + counts.T)/2
    
    dist_flatten = true_dist[np.triu_indices(true_dist.shape[0], k=1)]
    avg_x = counts.values[np.triu_indices(counts.shape[0], k=1)]
    avg_x = 101*avg_x/(100*avg_x+1)
    means = binned_statistic(avg_x, dist_flatten, 
                             statistic='mean', 
                             bins=100, 
                             range=(0, 1.0))
    means_y = np.nan_to_num(means[0], nan=1)
    means_x = means[1]
    bins_mid = np.array([means_x[i] for i in range(len(means_x)-1)])
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


def cluster_beads(counts_sp, gbr):
    G = igraph.Graph.Weighted_Adjacency(counts_sp)
    bead_idx = np.array(list(range(counts_sp.shape[0])))
    partition2 = la.find_partition(G, la.ModularityVertexPartition, weights="weight", max_comm_size=2500)
    clusters2 = partition2._membership
    counter = dict(Counter(clusters2))

    # loop
    loop_bool = True
    count = 1
    overlap_thresh = 30
    cluster_thresh = 200
    
    while loop_bool:
        counter = dict(Counter(clusters2))
        counter = pd.Series(counter)
        counter.sort_values(inplace=True)
        print(count)
        ################
        from collections import defaultdict
    
        # Initialize a defaultdict to store boundary nodes with keys as the top bordering cluster
        boundary_nodes = defaultdict(list)
        
        # Threshold for the ratio of edges connecting to nodes in different clusters
        threshold = 0.3
        
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
        sparse_df = pd.DataFrame.sparse.from_spmatrix(counts_sp)
        bead_idx = bead_idx[indices]
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
                print(i,counter[i])
                keys = [j for j in overlapping.keys() if j[0]==i or j[1]==i]
                #overlaps = {l:len(overlapping[l]) for l in keys}
        
                # merge with smallest neighboring cluster
                counter2 = dict(Counter(clusters2))
                counter2 = pd.Series(counter2)
                not_yet = list(set(flatten(keys))-set([i]))
                list_sizes = [counter2[t] for t in not_yet]
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
        sparse_df = pd.DataFrame.sparse.from_spmatrix(counts_sp)
        bead_idx = bead_idx[indices]
        ###################
        from collections import defaultdict
    
        # Initialize a defaultdict to store boundary nodes with keys as the top bordering cluster
        boundary_nodes = defaultdict(list)
        
        # Threshold for the ratio of edges connecting to nodes in different clusters
        threshold = 0.3
        
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
        sparse_df = pd.DataFrame.sparse.from_spmatrix(counts_sp)
        bead_idx = bead_idx[indices]
        ##################
        count += 1

    # end loop
    return counts_sp, clusters2, bead_idx, cluster_dict, total_dict, overlapping, boundary_nodes, sparse_df


def cluster_tsne(counts_sp, gbr, clusters2, bead_idx, cluster_dict, total_dict, overlapping, boundary_nodes, sparse_df, dir):
    tsne_dict = {}
    N = len(np.unique(clusters2))
    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    color_dict = pd.Series(dict(zip(range(len(clusters2)),clusters2)))

    overlap_thresh = 30
    
    def subset_has_boundary_neighbors_spiky_alpha(points, r, boundary_points_indices, proportion_threshold, alpha=1.0):
        """
        Check if a specified proportion of the points in each subset have neighbors on the boundary of the alpha shape.
    
        Args:
        - points: A 2D numpy array where each row represents a point with its x and y coordinates.
        - k: The number of nearest neighbors to consider.
        - boundary_points_indices: A list of lists containing indices of points on the boundary for each subset.
        - proportion_threshold: The minimum proportion of points in each subset that should have neighbors on the boundary.
        - alpha: The alpha value to control the spikiness of the alpha shape. Default is 0.1.
    
        Returns:
        - has_boundary_neighbors: A list of boolean values indicating whether the proportion of points with boundary 
        neighbors in each subset meets the threshold.
        """
        noise = 1e-8 * np.random.rand(*points.shape)
        points += noise
    
        def median_dist(mat):
            #mat = np.multiply(mat.values, vars)
            noise = 1e-8 * np.random.rand(*mat.shape)
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
        
        avg_dist = median_dist(points)
        
        tree = KDTree(points)
        
        alpha_shape = alphashape(points, alpha)
        boundary_line = alpha_shape.boundary
        
        has_boundary_neighbors = []
        prop = []
        for boundary_indices in boundary_points_indices:
            if len(boundary_indices) >= overlap_thresh:
                subset_boundary_count = 0
                for point_index in boundary_indices:
                    #_, neighbor_indices = tree.query(points[point_index].reshape(1, -1), k=k+1)
                    #print(neighbor_indices)
                    neighbor_indices = tree.query_ball_point(points[point_index], r=avg_dist*r, workers=-1)
                    neighbor_indices = np.array(neighbor_indices)
                    #print(neighbor_indices)
        
                    try:
                        neighbor_indices = neighbor_indices.squeeze()[1:]  # Exclude the point itself
            
                        # Convert neighbor points to Shapely Point objects
                        neighbor_points = [Point(points[neighbor_index]) for neighbor_index in neighbor_indices]
                        
                        # Check if any neighbor lies on the boundary of the alpha shape
                        if any(point.intersects(boundary_line) for point in neighbor_points):
                            subset_boundary_count += 1
                    except:
                        subset_boundary_count += 0
                
                proportion_boundary_points = subset_boundary_count / len(boundary_indices)
                print(proportion_boundary_points, len(boundary_indices))
                prop.append(proportion_boundary_points)
                
                if proportion_boundary_points >= proportion_threshold:
                    has_boundary_neighbors.append(True)
                else:
                    has_boundary_neighbors.append(False)
    
        return has_boundary_neighbors, prop

    def stdev_cluster(mat):
        #mat = mat.values#np.multiply(mat.values, vars)
        noise = 1e-8 * np.random.rand(*mat.shape)
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
        stdev = np.std(edge_lengths)
        return stdev/np.mean(edge_lengths)

    def point_cloud_to_binary_image(point_cloud):
        # Compute the average Delaunay triangulation distance
        #tri = Delaunay(point_cloud)
    
        def median_dist(mat):
            #mat = np.multiply(mat.values, vars)
            noise = 1e-8 * np.random.rand(*mat.shape)
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
    
        grid_size = 1.0 * avg_dist
    
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
        eroded_image = binary_erosion(binary_image, np.ones((5,5)))
        eroded_image = binary_erosion(eroded_image, np.ones((5,5)))
    
        # Perform watershed segmentation
        segmented_image = chan_vese(eroded_image)#distance, markers, mask=eroded_image)
        
        # Analyze the segmented image
        labeled_image, num_objects = label(segmented_image, connectivity=1, return_num=True)
        object_sizes = [np.sum(labeled_image == label) for label in range(1, num_objects + 1)]
    
        return num_objects, object_sizes
    
    def assess_even_spacing_histogram(points, num_bins=10):
        """
        Assess whether a set of points are evenly spaced by checking the uniformity of a 2D histogram.
        
        Args:
            points (np.ndarray): An array of shape (N, 2) representing N points in 2D space.
            num_bins (int): The number of bins for the 2D histogram in each dimension.
            
        Returns:
            bool: True if the points are evenly spaced, False otherwise.
            float: The chi-square statistic for the uniformity test.
            np.ndarray: The 2D histogram counts.
        """
        # Compute the 2D histogram
        H, xedges, yedges = np.histogram2d(points[:, 0], points[:, 1], bins=num_bins, 
                                           range=((np.min(points[:, 0]), np.max(points[:, 0])), (np.min(points[:, 1]), np.max(points[:, 1]))))
        # Compute the expected uniform count
        expected_count = np.sum(H) / (num_bins ** 2)
        # Compute the chi-square statistic
        chi_square = np.sum((H - expected_count) ** 2 / expected_count)

        return chi_square, H

    ######################################################
    
    for i in total_dict.keys():
        mpl.rcParams['font.size'] = 8
        mpl.rcParams['figure.dpi'] = 200
        
        print("Cluster {}".format(i))
        #green_intersect = list(set(green_points).intersection(set(total_dict[i])))
        subset = sparse_df.loc[total_dict[i],total_dict[i]]
        subset = subset.values#np.array(subset)#.toarray()
    
        avg_x = subset[np.triu_indices(subset.shape[0], k=1)]
        avg_x = 101*avg_x/(100*avg_x+1)
        
        dist_pred = gbr.predict(avg_x.reshape(-1, 1))
        dist_mat = np.zeros(subset.shape)
        inds = np.triu_indices_from(dist_mat, k = 1)
        dist_mat[inds] = dist_pred
        dist_mat[(inds[1], inds[0])] = dist_pred
        np.fill_diagonal(dist_mat, 0)
        
        print("Computing spectral embedding initialization...")
        time1 = timeit.default_timer()
        se = SpectralEmbedding(affinity="precomputed")
        se1 = se.fit_transform(subset)
        time2 = timeit.default_timer()
        print("Time to compute spectral embedding: {}\n".format(datetime.timedelta(seconds=int(time2-time1))))
    
        print("Computing TSNE...")
        time1 = timeit.default_timer()
    
        ###############
        def run_tsne(perp):
            tsne = TSNE(metric='precomputed', init=se1,#init="random", 
                        perplexity=perp, learning_rate="auto", early_exaggeration=12.0,
                       angle=1.0, method="barnes_hut", 
                        n_iter=500, n_iter_without_progress=50, min_grad_norm=1e-7)
            tsne1 = tsne.fit_transform(dist_mat)
            return tsne1
        perp_range = [50,100,150,200,250,300,400,500,750,1000,1250,1500]
        perp_range = [k for k in perp_range if k<len(total_dict[i])]
        results = Parallel(n_jobs=-1)(delayed(run_tsne)(i) for i in perp_range)
        results = np.array(list(results))
    
        fig, axs = plt.subplots(3, 4, figsize=(12, 9))
        axs = axs.flatten()
        for k in range(len(results)):
            ax = axs[k]
            ax.scatter(results[k][:,0], results[k][:,1], s=2, 
                c=color_dict[total_dict[i]], cmap=cmap, edgecolors='black', linewidth=0.1, label="TSNE")
            #plt.scatter(results[k][green_intersect,0], results[k][green_intersect,1], s=10, color="red", marker="+", linewidths=0.5)
            #ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax.set_title("Cluster {}, Perp={}".format(i,perp_range[k]))
            #plt.show()
            
            #plt.tight_layout()
            #plt.savefig(dir+"/initial_tsne.png", dpi=300)
            #plt.show(
            #plt.subplot(1,9,k+1)
        plt.tight_layout()
        plt.savefig(dir+"tsne_grid_{}.svg".format(i), dpi=500)
        plt.show()
        
        bool_array = []
        prop_array = []
        obj_bool = []
        
        print("*****")
        for j in range(len(results)):
            print("Cluster: {}, Perp={}".format(i,perp_range[j]))
            '''
            label_counts = dict(Counter(color_dict[total_dict[i]].values))
            del label_counts[i]
            index_dict = {}
            for k in label_counts.keys():
                if label_counts[k] >= 0:#overlap_thresh:
                    index_dict[k] = np.where(color_dict[total_dict[i]]==k)[0]
            '''
    
            index_dict = {}
    
            neighbors = [l[1] for l in overlapping.keys() if i in l]
            neighbors = [l for l in neighbors if l!=i]
            neighbors = np.unique(neighbors)
            for m in neighbors:
                mask = np.isin(total_dict[i], overlapping[(i,m)])
                indices = np.nonzero(mask)[0]
                if len(indices) >= overlap_thresh:
                    index_dict[m] = indices
        
            points = results[j]#tsne_dict[j].values
            subsets = list(index_dict.values())
            
            is_on_boundary, prop = subset_has_boundary_neighbors_spiky_alpha(points, 5, subsets, 0.8)
            print(is_on_boundary)
            bool_array.append(all(is_on_boundary))
            print(bool_array)
            prop_array.append(np.min(prop))
            
            print("Is each subset on the boundary?")
            for k, result in enumerate(is_on_boundary):
                print(f"Subset {k+1}: {result}")
                
            print("Detecting number of objects...")
            binary_img = point_cloud_to_binary_image(points)
            num_objects, object_sizes = analyze_binary_image(binary_img)
            object_sizes = np.array(object_sizes)
            object_sizes = object_sizes[object_sizes>20]
            print(len(object_sizes))
            if len(object_sizes) == 1:
                obj_bool.append(True)
            else:
                obj_bool.append(False)
            
            print("---------------------------------")
        bool_array = np.array(bool_array)
        print("border bool: {}".format(bool_array))
        print("segmented bool: {}".format(obj_bool))
        orig_bool = bool_array.copy()
        
        bool_array = bool_array & obj_bool
        print("combined bool: {}".format(bool_array))
        
        new_results = results[bool_array]
        new_params = np.array(perp_range)[bool_array]
        print(new_params)
    
        #if len(new_params)==0:
        #    new_results = np.array(results[np.argmax(prop_array)])
        #    new_params = np.array(perp_range[np.argmax(prop_array)])
        if not any(bool_array):
            bool_array = obj_bool.copy()
            new_results = results[bool_array]
            new_params = np.array(perp_range)[bool_array]
            prop_array = np.array(prop_array)[bool_array]
            new_results = np.array([results[np.argmax(prop_array)]])
            new_params = np.array([perp_range[np.argmax(prop_array)]])
    
        print(new_params)
        print(new_results.shape)
        print(new_results[0].shape)
        #stdevs = [stdev_cluster(mat) for mat in new_results]
        stdevs = []
    
        n = len(new_results)  # Number of subplots
        cols = math.ceil(math.sqrt(n))  # Calculate number of columns
        rows = math.ceil(n / cols)
        fig, axs = plt.subplots(rows, cols, figsize=(10, 8))
        try:
            axs = axs.flatten()
        except:
            axs = [axs]
        
        for k in range(len(new_results)):
            mat = new_results[k]
            bins = int(mat.shape[0]/40)
            chi_square, H = assess_even_spacing_histogram(mat, bins)
            stdevs.append(chi_square)
            ax = axs[k]
            ax.imshow(H, origin='lower', cmap='viridis')
            ax.set_title("Perp={}".format(new_params[k]))
        plt.show()
    
        print(stdevs)
        
        tsne1 = new_results[np.argmin(stdevs)]
        
        ###############
        
        
        time2 = timeit.default_timer()
        print("Time to compute TSNE: {}\n".format(datetime.timedelta(seconds=int(time2-time1))))
    
        print("Accepted perplexity: {}".format(new_params[np.argmin(stdevs)]))
        tsne_df = pd.DataFrame(index=total_dict[i], data=tsne1)
        
        #plt.scatter(X_orig[total_dict[i],0], X_orig[total_dict[i],1], s=5, color="lightblue", 
        #            marker="x", linewidths=0.5, label="Original")
        #plt.scatter(X_orig[green_intersect,0], X_orig[green_intersect,1], s=8, color="green", marker="x", linewidths=0.5)
        plt.scatter(tsne_df.loc[:,0], tsne_df.loc[:,1], s=2, 
                c=color_dict[tsne_df.index], cmap=cmap, edgecolors='black', linewidth=0.1, label="TSNE")
        #plt.scatter(tsne_df.loc[green_intersect,0], tsne_df.loc[green_intersect,1], s=10, color="red", marker="+", linewidths=0.5)
        #plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.title("Cluster {}".format(i))
        plt.tight_layout()
        plt.savefig(dir+"/tsne_{}.png".format(i), dpi=500)
        plt.show()
    
        tsne_df = pd.DataFrame(index=total_dict[i], data=tsne1)
        tsne_dict[i] = tsne_df
        print("----------------------------")
    mpl.rcParams['font.size'] = 12
    mpl.rcParams['figure.dpi'] = 500


    def median_cluster(mat):
        #mat = mat.values#np.multiply(mat.values, vars)
        noise = 1e-8 * np.random.rand(*mat.shape)
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


    tsne_scaled = {}
    tsne_scaled_orig = {}
    
    for i in tsne_dict.keys():
        print(i)
        centroid = np.mean(tsne_dict[i], axis=0)
        median = median_cluster(tsne_dict[i].values)
        scaling_factor = 1/median
        tsne_scaled[i] = centroid + scaling_factor * (tsne_dict[i] - centroid)
        tsne_scaled_orig[i] = tsne_scaled[i].loc[cluster_dict[i],:]
        print("----")


    return overlapping, tsne_scaled, tsne_scaled_orig, clusters2, bead_idx


def stitch_clusters(overlapping, tsne_scaled, tsne_scaled_orig, clusters2, bead_idx, unique, dir, sample_dir):
    def rigid_transform(vars, points, reflection_x=False, reflection_y=False):
        # Apply rotation
        rotation_angle, translation_x, translation_y, = vars
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
        points = np.dot(points, rotation_matrix)
        
        # Apply translation
        translation_vector = np.array([translation_x,translation_y])
        points += translation_vector
    
        ref_mat = np.array([[1,0],[0,1]])
        if reflection_x:
            ref_mat[1,1] = -1
        if reflection_y:
            ref_mat[0,0] = -1
    
        points = np.dot(points, ref_mat)
        
        return points

    def rotate(vars, points, reflection_x=False, reflection_y=False):
        # Apply rotation
        rotation_angle, translation_x, translation_y, = vars
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
        points = np.dot(points, rotation_matrix)
        
        # Apply translation
        #translation_vector = np.array([translation_x,translation_y])
        #points += translation_vector
    
        ref_mat = np.array([[1,0],[0,1]])
        if reflection_x:
            ref_mat[1,1] = -1
        if reflection_y:
            ref_mat[0,0] = -1
    
        points = np.dot(points, ref_mat)
        
        return points

    def matrix_error(vars, mat1, mat2, indices, reflection_x, reflection_y):
        mat1_orig = mat1.values.copy()
        mat2_orig = mat2.values.copy()
        mat1 = mat1.loc[indices,:].values
        mat2 = mat2.loc[indices,:].values
        
        mat2_transf = rigid_transform(vars, mat2, reflection_x, reflection_y)
    
        mat2_orig_transf = rigid_transform(vars, mat2_orig, reflection_x, reflection_y)
        
        diff = mat2_transf-mat1
        diff = np.multiply(diff, diff)
        #diff = diff.T
        #print(diff.shape)
        diff = diff.sum(axis=1)
        diff = np.sqrt(diff)
        diff = np.mean(diff)
        
        return diff

    N = len(np.unique(clusters2))
    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    color_dict = pd.Series(dict(zip(range(len(clusters2)),clusters2)))
    len_dict = {}
    for i in overlapping.keys():
        len_dict[i[0]] = len(overlapping[i])
        len_dict[i[1]] = len(overlapping[i])
    max_key = max(len_dict, key=len_dict.get)
    def flatten(xss):
        return [x for xs in xss for x in xs]

    ############################################################
    #stitching the puzzle pieces

    list_added = [max_key]
    tsne_stitch = tsne_scaled[max_key]
    
    puzzle_transf = {}
    bool_dict = {}
    
    new_dict = {}
    new_dict[max_key] = tsne_scaled[max_key]
    
    figs = []
    
    fig, ax = plt.subplots()
    ax.scatter(tsne_stitch.loc[:,0], tsne_stitch.loc[:,1], s=2, 
            c=color_dict[tsne_stitch.index], cmap=cmap, edgecolors='black', linewidth=0.1, label="TSNE")
    #ax.set_xlim(-100, 150)
    #ax.set_ylim(-60, 210)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Puzzle Solution")
    fig.canvas.draw()
    figs.append(fig)
    plt.savefig(dir+"/puzzle_stitch_{}.svg".format(len(list_added)), dpi=500)
    plt.pause(2)
    ax.clear()
    
    #fig, ax = plt.subplots()
    
    while len(list_added) != len(tsne_scaled):
        time1 = timeit.default_timer()
        keys = [i for i in overlapping.keys() if i[0] in list_added or i[1] in list_added]
        not_yet = list(set(flatten(keys))-set(list_added))
        
        # add the next piece with the most overlapping points to the current stitch
        indices_list = []
        indices_len = []
        for j in range(len(not_yet)):
            new = not_yet[j]
            keys_new = [i for i in keys if new in i]
            #print(keys)
            mat2 = tsne_scaled[new]
            indices = []
            for i in keys_new:
                indices += overlapping[i]
            indices = np.unique(indices)
            indices_list.append(indices)
            indices_len.append(len(indices))
    
        idx = np.argmax(indices_len)
        indices = indices_list[idx]
        mat2 = tsne_scaled[not_yet[idx]]
        print(not_yet)
        print(indices_len)
        print(idx)
        print("New cluster: {}".format(not_yet[idx]))
        #transl = np.mean(tsne_stitch.loc[indices,:].values-mat2.loc[indices,:].values, axis=0)
        new = not_yet[idx]
        
    
        #new = not_yet[0]
        print("New cluster: {}".format(new))
        keys = [i for i in keys if new in i]
        print(keys)
        mat2 = tsne_scaled[new]
        indices = []
        for i in keys:
            indices += overlapping[i]
        indices = np.unique(indices)
    
        
        results_list = []
        fmin = []
        bool_list = [[False,False],[False,True],[True,True],[True,False]]
        angle_grid = [0]#[2*np.pi/i for i in range(1,9)]
        param_list = []
        class_bool = []
        score_list = []
        
        for i in bool_list:
            for j in angle_grid:
                rt = rotate(np.array([j,0,0]), mat2.loc[indices,:].values, i[0], i[1])
                translation = (tsne_stitch.loc[indices,:].values - rt)#.flatten()
                #print(translation.shape)
                translation = np.mean(translation, axis=0)
                #print(translation.shape)
    
                res_tsne = basinhopping(matrix_error, np.array([j,translation[0],translation[1]]), niter=10,
                             minimizer_kwargs={"args":(tsne_stitch, mat2, indices, i[0], i[1]),
                                     'method':"Powell"},
                             stepsize=1.0, T=1.0, disp=False)#, accept_test=accept_test)
                results_list.append(res_tsne)
                fmin.append(res_tsne.fun)
                param_list.append(i)
                print(i,j,res_tsne.fun)
    
                ###
                mat2_orig_transf = rigid_transform(res_tsne.x, mat2.values, i[0], i[1])
                mat1_orig = tsne_stitch.values
                LR = LogisticRegression()
                LR.fit(np.vstack([mat1_orig, mat2_orig_transf]), y=np.array([0]*mat1_orig.shape[0]+[1]*mat2_orig_transf.shape[0]))
                pred = LR.predict(np.vstack([mat1_orig, mat2_orig_transf]))
                score = precision_score(np.array([0]*mat1_orig.shape[0]+[1]*mat2_orig_transf.shape[0]), pred)
                score_list.append(score)
                if score>=0.6:
                    class_bool.append(True)
                else:
                    class_bool.append(False)
                ###
    
        print(score_list)
        print(class_bool)
        ########
        class_bool = np.array(class_bool)
        score_list = np.array(score_list)
        class_bool = (score_list == np.max(score_list))
        fmin = np.array(fmin)[class_bool]
        results_list = np.array(results_list)[class_bool]
        param_list = np.array(param_list)[class_bool]
        
    
        min_f = np.argmin(fmin)
        res_tsne = results_list[min_f]
        
        puzzle_transf[new] = res_tsne.x
        bool_dict[new] = param_list[min_f]
        
        tsne_new = pd.DataFrame(data=rigid_transform(res_tsne.x, tsne_scaled[new], param_list[min_f][0], param_list[min_f][1]), 
                                index=tsne_scaled[new].index)
    
        new_dict[new] = tsne_new
        
        list_added.append(new)
        tsne_stitch = pd.concat([tsne_stitch, tsne_new])
        tsne_stitch = tsne_stitch.groupby(level=0).mean()
        print(tsne_stitch.shape)
    
        fig, ax = plt.subplots()
        ax.scatter(tsne_stitch.loc[:,0], tsne_stitch.loc[:,1], s=2, 
                c=color_dict[tsne_stitch.index], cmap=cmap, edgecolors='black', linewidth=0.1, label="TSNE")
        #ax.set_xlim(-100, 150)
        #ax.set_ylim(-60, 210)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.set_title("Puzzle Solution")
        fig.canvas.draw()
        figs.append(fig)
        plt.savefig(dir+"/puzzle_stitch_{}.svg".format(len(list_added)), dpi=500)
        plt.pause(2)
        
        if len(list_added) < len(tsne_scaled):
            ax.clear()
        
        #tsne_new = matrix_transform(res_tsne.x, mat2)
        time2 = timeit.default_timer()
        print("Time to stitch: {}\n".format(datetime.timedelta(seconds=int(time2-time1))))
        #df = pd.DataFrame(data=matrix_transform(res_tsne.x[i*8:i*8+8], tsne_dict[i].loc[cluster_dict[i],:]), index=cluster_dict[i])
    plt.close(fig)
    #############################################

    tsne_stitch = tsne_scaled_orig[max_key]
    for i in puzzle_transf.keys():
        tsne_new = pd.DataFrame(data=rigid_transform(puzzle_transf[i], tsne_scaled_orig[i], bool_dict[i][0], bool_dict[i][1]), 
                                index=tsne_scaled_orig[i].index)
        tsne_stitch = pd.concat([tsne_stitch, tsne_new])
        tsne_stitch = tsne_stitch.groupby(level=0).mean()

    tsne_stitch['X0'] = tsne_stitch[0].copy()
    del tsne_stitch[0]
    tsne_stitch['X1'] = tsne_stitch[1].copy()
    del tsne_stitch[1]

    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    plt.scatter(tsne_stitch.loc[:,'X0'], tsne_stitch.loc[:,'X1'], s=2, cmap=cmap, edgecolors='black', linewidth=0.1, c=clusters2)#c=clusters2
    plt.colorbar()
    #plt.xlim(-100, 150)
    #plt.ylim(-60, 210)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Initial Puzzle Solution ")
    plt.savefig(dir+"/puzzle_solution_initial.svg", dpi=500)
    plt.show()


    tsne_stitch['cluster'] = clusters2
    tsne_stitch.index = bead_idx
    tsne_stitch.to_csv("{}initial_solution.csv".format(sample_dir))

    tsne_barcodes = tsne_stitch.copy()
    mapping_rev = dict(zip(list(range(len(unique))), unique))
    tsne_barcodes.index = tsne_barcodes.index.map(mapping_rev)
    tsne_barcodes.to_csv("{}initial_solution_barcodes.csv".format(sample_dir))

    return dir, sample_dir, new_dict, overlapping, tsne_scaled_orig, tsne_scaled, bead_idx, N, unique, tsne_stitch
    

def refine_solution(dir, sample_dir, new_dict, overlapping, tsne_scaled_orig, tsne_scaled, bead_idx, N, unique, tsne_stitch):
    def matrix_error(vars1, vars2, mat1, mat2, indices):#, error_bool=False):
        rotation_angle, translation_x, translation_y = vars1
        mat1 = mat1.loc[indices,:].values
        mat2 = mat2.loc[indices,:].values
    
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
        
        #centroid1 = np.mean(mat1, axis=0)
        #mat1_transf = centroid1 + scale1 * (mat1 - centroid1)
        
        mat1_transf = np.dot(mat1, rotation_matrix)
        translation_vector = np.array([translation_x,translation_y])
        mat1_transf += translation_vector
    
    
        rotation_angle, translation_x, translation_y = vars2
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
    
        #centroid2 = np.mean(mat2, axis=0)
        #mat2_transf = centroid2 + scale2 * (mat2 - centroid2)
        
        mat2_transf = np.dot(mat2, rotation_matrix)
        translation_vector = np.array([translation_x,translation_y])
        mat2_transf += translation_vector
         
        #error = np.linalg.norm(mat2_transf-mat1.T, ord="fro")
        diff = mat2_transf-mat1_transf
        diff = np.multiply(diff, diff)
        diff = diff.T
        #print(diff.shape)
        diff = diff.sum(axis=1)
        diff = np.sqrt(diff)
        diff = np.mean(diff)
        return diff

    def pairwise_error(vars, tsne_scaled, overlapping):
        #res_dict = {}
        error = 0
        pairs_calc = []
        for pair in overlapping.keys():
            #print(pair)
            i, j = pair
            var_i = vars[i*3:i*3+3]
            #print(var_i)
            var_j = vars[j*3:j*3+3]
            #print(var_j)
            if (j,i) not in pairs_calc:
                error += matrix_error(var_i, var_j, tsne_scaled[i], tsne_scaled[j], overlapping[(i,j)])
                pairs_calc.append(pair)
            #print(error)
            #res_dict[i] =  rigid_transform(var_i, tsne_scaled[i])
            #res_dict[j] =  rigid_transform(var_j, tsne_scaled[j])
        return error#, res_dict

    def rigid_transform(vars, points, reflection_x=False, reflection_y=False):
        #points = points.values
        # Apply rotation
        rotation_angle, translation_x, translation_y = vars
        rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                    [np.sin(rotation_angle), np.cos(rotation_angle)]])
    
        #centroid1 = np.mean(points, axis=0)
        #points = centroid1 + scale1 * (points - centroid1)
        
        points = np.dot(points, rotation_matrix)
        
        # Apply translation
        translation_vector = np.array([translation_x,translation_y])
        points += translation_vector
    
        ref_mat = np.array([[1,0],[0,1]])
        if reflection_x:
            ref_mat[1,1] = -1
        if reflection_y:
            ref_mat[0,0] = -1
    
        points = np.dot(points, ref_mat)
        return points

    print("Refining solution...")
    time1 = timeit.default_timer()
    #error_bool = False
    res_tsne = minimize(pairwise_error, np.array([0,0,0]*(max(new_dict.keys())+1)), 
                   args=(new_dict, overlapping), method="L-BFGS-B", options={"disp":True})
    time2 = timeit.default_timer()
    print("Time for refinement: {}\n".format(datetime.timedelta(seconds=int(time2-time1))))

    
    transf_list = []
    test_dict = {}
    for i in tsne_scaled_orig.keys():
        df = pd.DataFrame(data=rigid_transform(res_tsne.x[i*3:i*3+3], new_dict[i].loc[tsne_scaled_orig[i].index,:]),
                         index=tsne_scaled_orig[i].index)
        df_overlap = pd.DataFrame(data=rigid_transform(res_tsne.x[i*3:i*3+3], new_dict[i].loc[tsne_scaled[i].index,:]),
                         index=tsne_scaled[i].index)
        test_dict[i] = df_overlap
        transf_list.append(df)

    
    tsne_total = pd.concat(transf_list)
    tsne_total = tsne_total.groupby(level=0).mean()
    tsne_total.index = bead_idx
    
    error = pairwise_error(np.array([0,0,0]*(max(new_dict.keys())+1)), new_dict, overlapping)
    print("Initial error: {}".format(error))
    error = pairwise_error(res_tsne.x, new_dict, overlapping)
    print("Final error: {}".format(error))

    tsne_total['X0'] = tsne_total[0].copy()
    del tsne_total[0]
    tsne_total['X1'] = tsne_total[1].copy()
    del tsne_total[1]

    plt.gcf()
    plt.clf()
    cmap = mpl.colors.ListedColormap(plt.get_cmap("gist_ncar")(np.linspace(0,1,N)))
    plt.scatter(tsne_total.loc[:,'X0'], tsne_total.loc[:,'X1'], s=2, c=clusters2, cmap=cmap, edgecolors='black', linewidth=0.1)
    plt.colorbar()
    #plt.xlim(-100, 150)
    #plt.ylim(-60, 210)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Final Puzzle Solution")
    plt.savefig(dir+"/puzzle_solution_final.svg", dpi=500)
    plt.show()

    
    tsne_total['cluster'] = tsne_stitch['cluster']
    tsne_total.to_csv("{}final_solution.csv".format(sample_dir))

    tsne_barcodes = tsne_total.copy()
    mapping_rev = dict(zip(list(range(len(unique))), unique))
    tsne_barcodes.index = tsne_barcodes.index.map(mapping_rev)
    tsne_barcodes.to_csv("{}final_solution.csv".format(sample_dir))

    return tsne_barcodes, tsne_total


def main_real():
    time_start = timeit.default_timer()

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument('-s', '--sample_id', help = "Sample ID", required = True, default = "")
    parser.add_argument('-i', '--interactions_path', help = "Path to interactions csv", required = True, default = "")
    
    #sample_id = ""
    argument = parser.parse_args()
    sample_id=str(argument.sample_id)
    int_path=str(argument.interactions_path)
    
    sample_dir = "./{}/".format(sample_id)
    os.mkdir(sample_dir)
    dir = "./{}/figures/".format(sample_id)
    os.mkdir(dir)
    
    counts_sp, unique = create_sparse_matrix_from_file(int_path)
    
    counts_sp, gbr = similarity_distance_mapping(dir, counts_sp)
    
    counts_sp, clusters2, bead_idx, cluster_dict, total_dict, overlapping, boundary_nodes, sparse_df = cluster_beads(counts_sp, gbr)

    overlapping, tsne_scaled, tsne_scaled_orig, clusters2, bead_idx = cluster_tsne(counts_sp, gbr, clusters2, bead_idx, cluster_dict, total_dict, overlapping, boundary_nodes, sparse_df, dir)

    dir, sample_dir, new_dict, overlapping, tsne_scaled_orig, tsne_scaled, bead_idx, N, unique, tsne_stitch = stitch_clusters(overlapping, tsne_scaled, tsne_scaled_orig, clusters2, bead_idx, unique, dir, sample_dir)

    tsne_barcodes, tsne_total = refine_solution(dir, sample_dir, new_dict, overlapping, tsne_scaled_orig, tsne_scaled, bead_idx, N, unique, tsne_stitch)

    time_end = timeit.default_timer()
    print("Total script time: {}".format(datetime.timedelta(seconds=int(time_end-time_start))))


def downsample(counts_sp, pct=1.0):
    adata = sc.AnnData(counts_sp)
    total_sum = adata.X.sum()
    adata_ds = sc.pp.downsample_counts(adata, total_counts=pct*total_sum, copy=True)
    counts_sp = csr_matrix(adata_ds.X)
    return counts_sp
    

def matching_error(coord_path, tsne_total, dir):
    print("Reading in coordinates...")
    coords = pd.read_csv(coord_path, index_col=0)
    X_orig = coords[['x_coord','y_coord']].values
    print(X_orig.shape)
    X_orig = X_orig[tsne_total.index,:].copy()
    green_points = np.random.choice(range(X_orig.shape[0]), size=100, replace=False)
    print(X_orig.shape)

    X_pred = tsne_total[['X0','X1']].values

    print("Computing linear transformation on fiducials...")
    time1 = timeit.default_timer()
    shuffle_order = np.random.permutation(range(len(green_points)))

    centroid_orig = np.mean(X_orig, axis=0)
    centroid_tsne = np.mean(X_pred, axis=0)
    transl = centroid_orig - centroid_tsne
    x_scale = (np.percentile(X_orig[:,0], 95)-np.percentile(X_orig[:,0], 5))/(np.percentile(X_pred[:,0], 95)-np.percentile(X_pred[:,0], 5))
    y_scale = (np.percentile(X_orig[:,1], 95)-np.percentile(X_orig[:,1], 5))/(np.percentile(X_pred[:,1], 95)-np.percentile(X_pred[:,1], 5))

    def matrix_transform(vars, mat):
        a, b, c, d, x, y, e, f = vars
        rotation = np.array([[a,b],[c,d]])
        translation = np.array([[x],[y]])
        scale = np.array([[e],[f]])
    
        mat_transf = np.dot(rotation, mat.T) + translation
        mat_transf = np.multiply(mat_transf, scale)
        return mat_transf.T
    
    def matrix_error(vars, mat1, mat2, error_bool=False):
        a, b, c, d, x, y, e, f = vars
        rotation = np.array([[a,b],[c,d]])
        translation = np.array([[x],[y]])
        scale = np.array([[e],[f]])
        N = mat1.shape[0]
    
        mat2_transf = np.dot(rotation, mat2.T) + translation
        mat2_transf = np.multiply(mat2_transf, scale)
    
        if error_bool:
            
            #mat1 = mat1.T
            C = pairwise_distances(mat1, mat2_transf.T, n_jobs=-1)
            row_ind, col_ind = linear_sum_assignment(C)
        
            mat1 = mat1[row_ind,:]
            mat2_transf = mat2_transf[:,col_ind]
        
            diff = mat2_transf-mat1.T
            diff = diff.T
            diff = np.multiply(diff, diff)
            diff = diff.sum(axis=1)
            diff = np.sqrt(diff)
            error = np.mean(diff)
            
        else:
            diff = mat2_transf-mat1.T
            diff = diff.T
            diff = np.multiply(diff, diff)
            diff = diff.sum(axis=1)
            diff = np.sqrt(diff)
            error = np.mean(diff)
        return error
    
    param_list = [[1,0,0,1,0,0,1,1],
              [1,0,0,1,transl[0],transl[1],x_scale,y_scale],
              [-1,0,0,1,transl[0],transl[1],x_scale,y_scale],
              [1,0,0,-1,transl[0],transl[1],x_scale,y_scale],
              [-1,0,0,-1,transl[0],transl[1],x_scale,y_scale],
              [0,1,1,0,transl[0],transl[1],x_scale,y_scale],
              [0,-1,1,0,transl[0],transl[1],x_scale,y_scale],
              [0,1,-1,0,transl[0],transl[1],x_scale,y_scale],
              [0,-1,-1,0,transl[0],transl[1],x_scale,y_scale]]
    error_list = []
    for i in param_list:
        error_list.append(matrix_error(i, X_orig[green_points,:], X_pred[green_points,:][shuffle_order,:], True))
    best = np.argmin(error_list)
    x0 = param_list[best]
    print(error_list)

    res_tsne = basinhopping(matrix_error, np.array(x0), niter=100,
               minimizer_kwargs={"args":(X_orig[green_points,:], X_pred[green_points,:][shuffle_order,:], True),#[shuffle_order,:]),
                                 'method':"Powell", 
                                 'bounds':[(-1.5,1.5),
                                           (-1.5,1.5),(-1.5,1.5),(-1.5,1.5),(None,None),(None,None),(None,None),(None,None)]},
                       stepsize=1.0, T=1.0, disp=True)
    time2 = timeit.default_timer()
    print("Time to transform fiducials: {}\n".format(datetime.timedelta(seconds=int(time2-time1))))

    shuffle_order = np.random.permutation(range(X_orig.shape[0]))
    X_shuffle = X_orig.copy() #X_orig[shuffle_order, :].copy()
    tsne_shuffle = X_pred[shuffle_order, :].copy()
    
    gp = np.zeros(X_orig.shape[0])
    gp[green_points] = 1
    gp = gp[shuffle_order].copy()
    green_shuffle = np.where(gp==1)[0]
    
    tsne_transf = matrix_transform(res_tsne.x, tsne_shuffle)

    plt.scatter(X_shuffle[:,0], X_shuffle[:,1], s=5, color="lightblue", 
            marker="x", linewidths=0.5, label="Original")
    plt.scatter(X_shuffle[green_points,0], X_shuffle[green_points,1], s=8, 
                color="green", marker="x", linewidths=0.5)
    plt.scatter(tsne_transf[:,0], tsne_transf[:,1], s=5, marker = "+", 
                linewidths=0.5, color="grey", label="TSNE")
    plt.scatter(tsne_transf[green_shuffle,0], tsne_transf[green_shuffle,1], s=8, color="red", 
                marker="+", linewidths=0.5)
    
    diff = X_orig-tsne_transf[np.argsort(shuffle_order),:]
    diff = np.multiply(diff, diff)
    diff = diff.sum(axis=1)
    diff = np.sqrt(diff)
    error = np.mean(diff)
    
    plt.title("Transformation based on fiducials\nMAE on fids={}, true on all ={}".format(
        round(res_tsne.fun,5), round(error,5)))
    plt.legend()
    plt.tight_layout()
    plt.savefig(dir+"/fiducial_alignment.png", dpi=300)
    plt.clf()

    ##################
    print("Computing initial point matching...")
    time1 = timeit.default_timer()
    C = pairwise_distances(X_shuffle, tsne_transf, n_jobs=-1)
    row_ind, col_ind = linear_sum_assignment(C)
    time2 = timeit.default_timer()
    print("Time to compute initial matching: {}\n".format(datetime.timedelta(seconds=int(time2-time1))))

    X_new = X_shuffle[row_ind,:]
    tsne_new = tsne_transf[col_ind,:]
    y_order = shuffle_order[col_ind]
    x_order = np.array(list(range(X_orig.shape[0])))[row_ind]

    plt.scatter(X_shuffle[:,0], X_shuffle[:,1], s=5, color="lightblue", 
            marker="x", linewidths=0.5, label="Original")
    plt.scatter(X_shuffle[green_points,0], X_shuffle[green_points,1], s=8, 
                color="green", marker="x", linewidths=0.5)
    plt.scatter(tsne_transf[:,0], tsne_transf[:,1], s=5, marker = "+", 
                linewidths=0.5, color="grey", label="TSNE")
    plt.scatter(tsne_transf[green_shuffle,0], tsne_transf[green_shuffle,1], s=8, color="red", 
                marker="+", linewidths=0.5)
    
    diff = X_new-tsne_new
    diff = np.multiply(diff, diff)
    diff = diff.sum(axis=1)
    diff = np.sqrt(diff)
    error = np.mean(diff)
    
    plt.title("Point matching MAE={}".format(round(error,5)))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(dir+"/initial_matching.png", dpi=300)
    plt.clf()
    ##############

    print("Computing in-place linear transform...")
    time1 = timeit.default_timer()
    res_tsne = minimize(matrix_error, np.array([1,0,0,1,0,0,1,1]), 
                   args=(X_new, tsne_new, False), method="Powell")
    time2 = timeit.default_timer()
    print("Time to compute linear transform: {}\n".format(datetime.timedelta(seconds=int(time2-time1))))
    
    tsne_transf = matrix_transform(res_tsne.x, tsne_new)
    
    plt.scatter(X_new[:,0], X_new[:,1], s=0.5, label="Original")
    plt.scatter(tsne_transf[:,0], tsne_transf[:,1], s=0.5, label="TSNE")
        
    for i in range(tsne_transf.shape[0]):
        plt.plot([X_new[i,0],tsne_transf[i,0]],[X_new[i,1],tsne_transf[i,1]], 
                 color="gray", linewidth=0.5)
    plt.title("Linear transform\nMAE={}".format(
        round(res_tsne.fun,5)))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(dir+"/linear_transform.png", dpi=300)
    plt.clf()

    ###############
    print("Recomputing point matching...")
    time1 = timeit.default_timer()
    C = pairwise_distances(X_new, tsne_transf, n_jobs=-1)
    row_ind, col_ind = linear_sum_assignment(C)
    time2 = timeit.default_timer()
    print("Time to recompute point matching: {}\n".format(datetime.timedelta(seconds=int(time2-time1))))

    y_order = y_order[col_ind]
    x_order = x_order[row_ind]
    
    X_new = X_new[row_ind,:].copy()
    tsne_new = tsne_transf[col_ind,:].copy()

    plt.scatter(X_new[:,0], X_new[:,1], s=0.5, label="Original")
    plt.scatter(tsne_new[:,0], tsne_new[:,1], s=0.5, label="TSNE")
    
    diff = X_new-tsne_new
    diff = np.multiply(diff, diff)
    diff = diff.sum(axis=1)
    diff = np.sqrt(diff)
    error = np.mean(diff)
    
    diff = X_new[np.argsort(x_order),:]-tsne_new[np.argsort(y_order),:]
    diff = np.multiply(diff, diff)
    diff = diff.sum(axis=1)
    diff = np.sqrt(diff)
    orig_error = np.mean(diff)
    
    for i in range(tsne_new.shape[0]):
        plt.plot([X_new[i,0],tsne_new[i,0]],[X_new[i,1],tsne_new[i,1]], 
                 color="gray", linewidth=0.5)
    plt.title("Redo point matching \nMatch MAE={}, orig MAE={}".format(
        round(error,5), round(orig_error,5)))
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(dir+"/point_matching.png", dpi=300)
    plt.clf()

    ##############
    X_final = X_new[np.argsort(y_order),:]
    TY_final = tsne_new[np.argsort(y_order),:]
    match_bool = (X_final==X_orig).all(axis=1)
    mismatch_idx = np.array(range(X_orig.shape[0]))[np.logical_not(match_bool)]

    diff = X_final-X_orig
    diff = np.multiply(diff, diff)
    diff = diff.sum(axis=1)
    print(diff.shape)
    diff = np.sqrt(diff)
    plt.hist(diff, bins=20, density=True)
    plt.xlabel("Inferred position error")
    plt.ylabel("Density")
    plt.savefig(dir+"/error_histogram.png", dpi=300)
    plt.clf()

    accuracy = (X_final==X_orig).all(axis=1).sum()/X_orig.shape[0]
    MAE = np.mean(diff)#np.linalg.norm(X_final-X_orig, ord="fro")/X_orig.shape[0]
    print("Final Accuracy: {}".format(accuracy))
    print("Final MAE: {}\n".format(MAE))

    plt.scatter(X_new[:,0], X_new[:,1], s=0.5, label="Original")
    plt.scatter(tsne_new[:,0], tsne_new[:,1], s=0.5, label="TSNE")
    #error = np.linalg.norm(X_new-TY, ord="fro")/X_new.shape[0]
    #orig_error = np.linalg.norm(X_new[np.argsort(x_order),:]-TY[np.argsort(y_order),:], 
    #                            ord="fro")/X_new.shape[0]
    
    for i in range(TY_final.shape[0]):
        if i in mismatch_idx:
            plt.plot([X_final[i,0],TY_final[i,0]],[X_final[i,1],TY_final[i,1]], 
                     color="red", linewidth=1)
            plt.plot([X_orig[i,0],TY_final[i,0]],[X_orig[i,1],TY_final[i,1]], 
                     color="green", linewidth=1)
            
    plt.title("Pairing mismatches (red incorrect, green correct)")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.tight_layout()
    plt.savefig(dir+"/pairing_mismatches.png", dpi=300)
    plt.clf()

    return MAE


def main_simulation():
    time_start = timeit.default_timer()

    parser = argparse.ArgumentParser(description = "Description for my parser")
    parser.add_argument('-s', '--sample_id', help = "Sample ID", required = True, default = "")
    parser.add_argument('-i', '--interactions_path', help = "Path to interactions csv", required = True, default = "")
    parser.add_argument('-c', '--coord_path', help = "Path to coordinates csv", required = True, default = "")
    parser.add_argument('-p', '--percent_downsample', help = "Downsample percentage", required = True, default = "")
    
    #sample_id = ""
    argument = parser.parse_args()
    sample_id=str(argument.sample_id)
    int_path=str(argument.interactions_path)
    coord_path=str(argument.coord_path)
    pct = float(argument.percent_downsample)
    
    sample_dir = "./{}/".format(sample_id)
    os.mkdir(sample_dir)
    dir = "./{}/figures/".format(sample_id)
    os.mkdir(dir)
    
    counts_sp, unique = create_sparse_matrix_from_file(int_path, "source_bead", "target_bead", "bead_counts")
    counts_sp = downsample(counts_sp, pct)

    counts_sp, gbr = similarity_distance_mapping(dir, counts_sp)
    
    counts_sp, clusters2, bead_idx, cluster_dict, total_dict, overlapping, boundary_nodes, sparse_df = cluster_beads(counts_sp, gbr)

    overlapping, tsne_scaled, tsne_scaled_orig, clusters2, bead_idx = cluster_tsne(counts_sp, gbr, clusters2, bead_idx, cluster_dict, total_dict, overlapping, boundary_nodes, sparse_df, dir)

    dir, sample_dir, new_dict, overlapping, tsne_scaled_orig, tsne_scaled, bead_idx, N, unique, tsne_stitch = stitch_clusters(overlapping, tsne_scaled, tsne_scaled_orig, clusters2, bead_idx, unique, dir, sample_dir)

    tsne_barcodes, tsne_total = refine_solution(dir, sample_dir, new_dict, overlapping, tsne_scaled_orig, tsne_scaled, bead_idx, N, unique, tsne_stitch)

    MAE = matching_error(coord_path, tsne_total, dir)

    time_end = timeit.default_timer()
    print("Total script time: {}".format(datetime.timedelta(seconds=int(time_end-time_start))))

    #
    return MAE
    


if __name__=="__main__":
    main_real()

    
