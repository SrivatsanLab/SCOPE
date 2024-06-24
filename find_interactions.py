import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import time
import json
import math
import sys




"""load de-duplicated library to csv"""
nextera_unique = pd.read_csv(sys.argv[1])


"""get list of all unique barcodes found in R1 and R2
"""
all_R1_full_bcs_series = nextera_unique["R1_full_bc"].value_counts()
all_R1_full_bcs = list(all_R1_full_bcs_series.keys())
print(f"R1 bcs: {len(all_R1_full_bcs)}")

all_R2_full_bcs_series = nextera_unique["R2_full_bc"].value_counts()
all_R2_full_bcs = list(all_R2_full_bcs_series.keys())
print(f"R2 bcs: {len(all_R2_full_bcs)}")


"""find only the intersection between R1 and R2 barcodes"""
intersection = list(set(all_R1_full_bcs) & set(all_R2_full_bcs))
print(f"intersection length: {len(intersection)}")


"""map each barcode to an index"""
bc_indices = dict()
for i in range(len(intersection)):
    bc = intersection[i]
    bc_indices[bc] = i

"""need to save dictionary!!"""

with open(sys.argv[2], "w") as f:
     f.write(json.dumps(bc_indices))



"""the barcode to index mapping has been saved, so we will load the dictionary from the same file every time"""

with open(sys.argv[2]) as json_file:
    barcode_indices = json.load(json_file)
    
print(f"# of bc indices: {len(barcode_indices)}")


"""filter the UMI-collapsed dataframe based on barcodes that appear in the intersection"""
"""
this filtering step actually doesn't change the size of the dataframe by much,
so we aren't losing too much information from the left-out barcodes 
"""

subset_nextera_unique = nextera_unique[nextera_unique['R1_full_bc'].isin(intersection)]
subset_nextera_unique = subset_nextera_unique[subset_nextera_unique['R2_full_bc'].isin(intersection)]
print(f"# of bc after filter for intersection: {len(subset_nextera_unique)}")

subset_nextera_unique.to_csv(sys.argv[3], index=False)

"""
Create new output file
Read dataframe with barcode strings line by line
Write the barcodes' integer IDs into a new line in the output file
Took around 80 seconds for a dataframe with 55612401 lines
"""

replaced_bc_file = open(sys.argv[4], "w")
replaced_bc_file.write("R1_bc,R2_bc\n")

start = time.time()
with open(sys.argv[3], "r") as f:
    next(f)
    for line in f:
        index, R1_full_bc, R1_UMI, sender, R2_UMI, R2_full_bc = line.rstrip().split(",")
        replaced_bc_file.write(f"{barcode_indices[R1_full_bc]},{barcode_indices[R2_full_bc]}\n")

end = time.time()
replaced_bc_file.close()  
print(end-start)



"""load dataframe with barcode-->integer associations"""

replaced_nextera_df = pd.read_csv(sys.argv[4], index_col=None)



nextera_interaction_df = replaced_nextera_df.groupby(["R1_bc", "R2_bc"]).size().reset_index().rename(columns={0:'count'})


nextera_interaction_df.to_csv(sys.argv[5], index=False)



"""

sys.argv[1]: deduplicated lib to load (csv)
sys.argv[2]: output json .txt file with barcode to numerical index matching
sys.argv[3]: subset the library to only include barcodes that appear in both R1 and R2 (.csv)
sys.argv[4]: the library's barcode strings have been replaced with numerical indices (.txt file)
sys.argv[5]: interactions.txt output filename

"""

