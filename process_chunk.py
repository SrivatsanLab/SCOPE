import gzip
import numpy as np
import sys
from Levenshtein import distance
from collections import defaultdict
import pandas as pd

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


SR_map = {SRs[i]: SRs[i+20] for i in range(20)}
SR_map.update({SRs[i+20]: SRs[i] for i in range(20)})
    

def pairwise_levenshtein_distances(string, vector_of_strings):
    """
    Calculate pairwise levenshtein distances between a string and a vector of strings.
    """
    return [distance(string, seq) for seq in vector_of_strings]

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
    # Initial intervals when extension is 0
    interval_1 = (0+overhang, 10+overhang)
    interval_2 = (14+overhang, 24+overhang)
    interval_3 = (28+overhang, 38+overhang)
    interval_4 = (42+overhang, 52+overhang)
    UMI_interval = (70+overhang, 76+overhang)
    SR_interval = (76+overhang, 96+overhang)

    # Determine extension
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
                # Sender is allowed to have at most 2 substitutions
                SR = SRs[np.argmin(SR_distances)]
                return corrected_bc + [UMI, SR]
            else:
                return ['3', corrected_bc[0], UMI] #if line in df has 3, SR did not align
        else:
            return ['2', '', ''] #if line in df has 2, bc did not align
    else:
        return ['1', '', ''] #if line in df has 1, scars did not align

def process_chunk(R1_filename, R2_filename, output_filename, dict_filename, start_sequence, num_sequences):
    # Initialize statistics
    scars_unaligned = 0
    bc_uncorrected = 0
    SR_uncorrected = 0
    reads_mapped = 0
    SR_distribution = defaultdict(lambda: [0] * 20)

    with gzip.open(R1_filename, "rt") as f1, gzip.open(R2_filename, "rt") as f2, open(output_filename, "w") as output_f:
        # Skip to the start sequence
        for _ in range(start_sequence * 4):
            f1.readline()
            f2.readline()

        # Process the chunk
        for _ in range(num_sequences):
            f1.readline()
            f2.readline()           
            seq1 = f1.readline().rstrip()
            seq2 = f2.readline().rstrip()
            
            seq1_data = get_barcodes(seq1, overhang=4)
            seq2_data = get_barcodes(seq2, overhang=4)
            f1.readline()
            f2.readline() 
            f1.readline()
            f2.readline() 

            if seq1_data[0] == '1' or seq2_data[0] == '1':
                scars_unaligned += 1
            elif seq1_data[0] == '2' or seq2_data[0] == '2':
                bc_uncorrected += 1
            elif seq1_data[0] == '3' or seq2_data[0] == '3':
                if seq1_data[0] != '3':
                    SR = seq1_data[2]
                    result_line = f"{seq1_data[0]}\t{seq1_data[1]}\t{SR}\t{seq2_data[2]}\t{seq2_data[1]}\n"
                    output_f.write(result_line)
                    SR_distribution[seq1_data[0]][SRs.index(SR)%20] += 1
                    SR_distribution[seq2_data[1]][SRs.index(SR)%20] += 1
                    reads_mapped += 1
                    
                elif seq2_data[0] != '3':
                    SR = seq2_data[2]
                    SR = SR_map[SR]
                    result_line = f"{seq1_data[1]}\t{seq1_data[2]}\t{SR}\t{seq2_data[1]}\t{seq2_data[0]}\n"
                    output_f.write(result_line)
                    SR_distribution[seq1_data[1]][SRs.index(SR)%20] += 1
                    SR_distribution[seq2_data[0]][SRs.index(SR)%20] += 1
                    reads_mapped += 1
                else:
                    SR_uncorrected += 1
            else:
                SR = seq1_data[2]
                result_line = f"{seq1_data[0]}\t{seq1_data[1]}\t{SR}\t{seq2_data[1]}\t{seq2_data[0]}\n"
                output_f.write(result_line)
                SR_distribution[seq1_data[0]][SRs.index(SR)%20] += 1
                SR_distribution[seq2_data[0]][SRs.index(SR)%20] += 1
                reads_mapped += 1
            
    
    print(f"Unaligned scars: {scars_unaligned}")
    print(f"Uncorrected bc: {bc_uncorrected}")
    print(f"Uncorrected SR: {SR_uncorrected}")
    print(f"Mapped reads: {reads_mapped}")
    print(len(SR_distribution))

    # Save SR_distribution to CSV
    df = pd.DataFrame.from_dict(SR_distribution, orient='index')
    df.to_csv(dict_filename, index=True, header=False)



if __name__ == "__main__":
    R1_filename = sys.argv[1]
    R2_filename = sys.argv[2]
    output_filename = sys.argv[3]
    dict_filename = sys.argv[4]
    start_sequence = int(sys.argv[5])
    num_sequences = int(sys.argv[6])
    
    process_chunk(R1_filename, R2_filename, output_filename, dict_filename, start_sequence, num_sequences)


