import time
import sys
import csv
import igraph as ig

def doublet_detection(input_file, threshold, output_file):
    """
    Input: a barcode interaction graph; a threshold that filters out barcodes with low degrees
    Output: 
    """

    print("Start loading interaction graph!")
    G = ig.Graph.Read_GraphML(input_file)

    start_time = time.time()
    print("Start detecting doublets!")
    doublets = []
    nodes = G.vs.indices
    for node in nodes:
        if len(G.neighbors(node)) > threshold:
            neighbors = G.neighbors(node)
            subgraph = G.induced_subgraph(neighbors)
            clusters = subgraph.community_leiden(objective_function="modularity", resolution_parameter=0.1)
            if len(clusters) > 1:
                sorted_clusters = sorted(clusters, key=len, reverse=True)
                if len(sorted_clusters[0]) < 4*len(sorted_clusters[1]):
                    doublets.append(node)

    print(f"{len(doublets):.4f} doublets detected!")              
    end_time = time.time()
    runtime = end_time - start_time  
    print(f"Runtime: {runtime:.4f} seconds")

    # Write doublets to file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(doublets)
    
    return doublets


doublet_detection(sys.argv[1], sys.argv[2], sys.argv[3])


"""
sys.argv[1]: path to the graph file .graphml
sys.argv[2]: thredhold for initial filtering (lower threshold, longer runtime, more doublets detected)
sys.argv[3]: path to output file .csv
"""