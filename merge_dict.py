import pandas as pd
from collections import defaultdict
import glob

# Function to read a CSV file and convert it to a dictionary
def read_csv_to_dict(filename):
    df = pd.read_csv(filename, header=None, index_col=0)
    return df.T.to_dict(orient='list')

# Initialize a defaultdict to store the merged results
merged_dict = defaultdict(lambda: [0] * 20)

# Get a list of all filenames matching the pattern 'dict_*.csv'
filenames = glob.glob("dict_*.csv")

# Read and merge each dictionary
for filename in filenames:
    current_dict = read_csv_to_dict(filename)
    for key, values in current_dict.items():
        for i, value in enumerate(values):
            merged_dict[key][i] += value

# Convert the merged dictionary to a regular dictionary (optional)
merged_dict = dict(merged_dict)

# Save dict to CSV
df = pd.DataFrame.from_dict(merged_dict, orient='index')
df.to_csv("merged_dict.csv", index=True, header=False)
