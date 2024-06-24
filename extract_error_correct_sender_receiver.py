import gzip
import numpy as np
import pandas as pd
import re
import time
import sys
from Levenshtein import distance


start_time = time.time()
print("start!")

"""load correct barcodes"""
BC_1s = list()
BC_2s = list()
BC_3s = list()
BC_4s = list()

# Read in barcodes library
with open("/fh/fast/srivatsan_s/pub/public_scripts/spatial_scripts/barcodes_only.txt", "r") as f:
    for line in f:
        BC, position = line.rstrip().split("\t")
        if position == "1":
            BC_1s.append(BC)
        elif position == "2":
            BC_2s.append(BC)
        elif position == "3":
            BC_3s.append(BC)
        elif position == "4":
            BC_4s.append(BC)


# Read in SR libraries
SRs = list()

with open("/fh/fast/srivatsan_s/pub/public_scripts/spatial_scripts/20240621_nupack_seq_in_use.txt", "r") as f:
    for line in f:
        SR = line.rstrip().split("\t")[0]
        SRs.append(SR)

print("Barcodes and SR sequences loaded")

def pairwise_levenshtein_distances(string, vector_of_strings):
    """
    Calculate pairwise levenshtein distances between a string and a vector of strings.
    """
    return [distance(string, seq) for seq in vector_of_strings]

# scars = "AACCACAGCCTATTCGAG"

def determine_extension(sequence, overhang): 
    # Create a list to store the sum of Levenshtein distances of all 4 scars given different possible lengths of extension   
    distances = [0, 0, 0, 0]
    distances[0] = distance(sequence[10+overhang:14+overhang], "AACC") + distance(sequence[24+overhang:28+overhang], "ACAG") + \
        distance(sequence[38+overhang:42+overhang], "CCTA") + distance(sequence[52+overhang:58+overhang], "TTCGAG") # If Length = 0
    distances[1] = distance(sequence[11+overhang:15+overhang], "AACC") + distance(sequence[25+overhang:29+overhang], "ACAG") + \
        distance(sequence[39+overhang:43+overhang], "CCTA") + distance(sequence[53+overhang:59+overhang], "TTCGAG") # If Length = 1
    distances[2] = distance(sequence[12+overhang:16+overhang], "AACC") + distance(sequence[26+overhang:30+overhang], "ACAG") + \
        distance(sequence[40+overhang:44+overhang], "CCTA") + distance(sequence[54+overhang:60+overhang], "TTCGAG") # If Length = 2
    distances[3] = distance(sequence[13+overhang:17+overhang], "AACC") + distance(sequence[27+overhang:31+overhang], "ACAG") + \
        distance(sequence[41+overhang:45+overhang], "CCTA") + distance(sequence[55+overhang:61+overhang], "TTCGAG") # If Length = 3
    
    min_distance = min(distances)
    if min_distance <= 2:
        # Take the extension length which makes the sum of distances to be at most 2
        extender = np.argmin(distances)
    else: 
        # If no such extension, the sequence is considered unaligned
        extender = None
    return extender


def get_barcodes(sequence, overhang):      
    # Initial inervals when extension is 0
    interval_1 = (0+overhang, 10+overhang)
    interval_2 = (14+overhang, 24+overhang)
    interval_3 = (28+overhang, 38+overhang)
    interval_4 = (42+overhang, 52+overhang)
    UMI_interval = (70+overhang, 76+overhang)
    SR_interval = (76+overhang, 96+overhang)

    # Determine extesion
    extension = determine_extension(sequence, overhang)

    if extension is not None:
        # If aligned, extract sequences for each barcode, UMI, and SR
        bc_1 = sequence[interval_1[0] + extension:interval_1[1] + extension]
        bc_2 = sequence[interval_2[0] + extension:interval_2[1] + extension]
        bc_3 = sequence[interval_3[0] + extension:interval_3[1] + extension]
        bc_4 = sequence[interval_4[0] + extension:interval_4[1] + extension]
        UMI = sequence[UMI_interval[0] + extension:UMI_interval[1] + extension]
        SR = sequence[SR_interval[0] + extension:SR_interval[1] + extension]

        bc1_distances = pairwise_levenshtein_distances(bc_1, BC_1s)
        bc2_distances = pairwise_levenshtein_distances(bc_2, BC_2s)
        bc3_distances = pairwise_levenshtein_distances(bc_3, BC_3s)
        bc4_distances = pairwise_levenshtein_distances(bc_4, BC_4s)

        if min(bc1_distances) <= 2 and min(bc2_distances) <= 2 and min(bc3_distances) <= 2 and min(bc4_distances) <= 2:
            # Each barcode is allowed to have at most 2 substitutions
            corrected_bc = [BC_1s[np.argmin(bc1_distances)] + BC_2s[np.argmin(bc2_distances)] + BC_3s[np.argmin(bc3_distances)] + \
                        BC_4s[np.argmin(bc4_distances)]]
                      
            SR_distances = pairwise_levenshtein_distances(SR, SRs)   
            min_SR_distance = min(SR_distances)  

            if min_SR_distance <= 2:
                # Sender is allowed to have at most 2 substituions
                SR = SRs[np.argmin(SR_distances)]
                return corrected_bc + [UMI, SR]
            else:
                return ['3', corrected_bc[0], UMI] #if line in df has 3, SR did not align
        else:
            return ['2', '', ''] #if line in df has 2, bc did not align
    else:
      return ['1', '', ''] #if line in df has 1, scars did not align



def get_sequences_into_df(R1_filename, R2_filename, output_filename):
    """writing into txt file, then loading as dataframe"""

    output_f = open(output_filename, "w")
    output_f.write("R1_full_bc\tR1_UMI\tSR\tR2_UMI\tR2_full_bc\n")
    with gzip.open(R1_filename, "rt") as f1, gzip.open(R2_filename, "rt") as f2:
        counter = 0
        for line_number, (line1, line2) in enumerate(zip(f1, f2)):
            # Print every second line out of 4
            if line_number % 4 == 1:
                seq1 = line1.rstrip()
                seq2 = line2.rstrip()
                seq1_data = get_barcodes(seq1, overhang= 4)
                seq2_data = get_barcodes(seq2, overhang= 4)
                SR = ''
                if seq1_data[0] == '1' or seq2_data[0] =='1':
                    # type 1 error, scars do not align, 'junk' read
                    output_f.write(f"{'1'}\t{''}\t{''}\t{''}\t{''}\n")
                elif seq1_data[0] == '2' or seq2_data[0] =='2':
                    # type 2 error, barcode doesnt match
                    output_f.write(f"{'2'}\t{''}\t{''}\t{''}\t{''}\n")
                elif seq1_data[0] == '3' or seq2_data[0] =='3':
                    # type 3 error, at least one SR doesnt match
                    if seq1_data[0] != '3':
                        # only SR from R2 doesnt match, take R1
                        SR = seq1_data[2]
                        output_f.write(f"{seq1_data[0]}\t{seq1_data[1]}\t{SR}\t{seq2_data[2]}\t{seq2_data[1]}\n")
                    elif seq2_data[0] != '3':
                        # only SR from R1 doesnt match, take R2
                        SR = seq2_data[2]
                        output_f.write(f"{seq1_data[1]}\t{seq1_data[2]}\t{SR}\t{seq2_data[1]}\t{seq2_data[0]}\n")
                    else:
                        # neither SR matches
                        output_f.write(f"{'3'}\t{''}\t{''}\t{''}\t{''}\n")        
                else:
                    # No error, take R1
                    SR = seq1_data[2]
                    output_f.write(f"{seq1_data[0]}\t{seq1_data[1]}\t{SR}\t{seq2_data[1]}\t{seq2_data[0]}\n")

            if counter % 1000000 == 0:
                print(f"Counter: {counter}")
            
            counter += 1

    print(f"# of reads in fastq: {counter/4}")

#fastq file to process barcodes
R1_filename = sys.argv[1]
R2_filename = sys.argv[2]

output_filename = sys.argv[3]



# #call function to write text file for loading dataframe
get_sequences_into_df(R1_filename, R2_filename, output_filename)


# this part is for dropping NAs from text file
# load df from text file
single_lib = pd.read_csv(sys.argv[3], sep="\t")

scars_unaligned_count = len(single_lib[single_lib["R1_full_bc"] == "1"])
bc_uncorrected = len(single_lib[single_lib["R1_full_bc"] == "2"])
sender_uncorrected = len(single_lib[single_lib["R1_full_bc"] == "3"])

print(f"Unaligned: {scars_unaligned_count}")
print(f"Uncorrected bc: {bc_uncorrected}")
print(f"Uncorrected sender: {sender_uncorrected}")


single_lib.replace("", float("NaN"), inplace=True)
single_lib.replace("1", float("NaN"), inplace=True)
single_lib.replace("2", float("NaN"), inplace=True)
single_lib.replace("3", float("NaN"), inplace=True)
single_lib.dropna(inplace=True)

print(f"truseq # of unique senders:")
print(single_lib['SR'].nunique())

single_lib.to_csv(sys.argv[4])

print(f"total number of reads: {len(single_lib)}")

end_time = time.time()
    
print(f"total time: {end_time - start_time}")



"""
sys.argv[1]: fastq file name R1
sys.argv[2]: fastq file name R2
# Default setting: overhang = 4
sys.argv[3]: write dataframe line by line to .txt file and to generate statistics
sys.argv[4]: after dropping NAs from dataframe, write final lib into csv file
"""
