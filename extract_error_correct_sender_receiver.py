import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
            


def reverse_complement(bases):
    """
    returns reverse complement of a sequence
    """
    DNA_dictionary = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
    rc_sequence = ""
    for base in reversed(bases):
        rc_sequence += DNA_dictionary[base]
    return rc_sequence


"""load new sender receiver sequences"""

SRs = list()

with open("/fh/fast/srivatsan_s/pub/public_scripts/spatial_scripts/nupack_seq.txt", "r") as f:
    for line in f:
        SR = line.rstrip().split("\t")[0]
        SRs.append(SR)


# append reverse complements of sender receiver sequences to SR list
rc_SRs = []  # Initialize an empty list to store reverse complements

for sr in SRs:
    rc_SRs.append(reverse_complement(sr))  # Append reverse complement of each sequence to rc_SRs


SRs.extend(rc_SRs)


print("bc sequences loaded")
print(len(SRs))

"""this function does not include the sequence itself in the output list"""
def find_one_off_errors(sequence):
    one_off_sequences = list()
    replace_base = {"A": ["C", "G", "T"], "C": ["A", "G", "T"], "G": ["A", "C", "T"], "T": ["A", "C", "G"]}
    for i in range(len(sequence)):
        replacements = replace_base[sequence[i]]
        for replacement in replacements:
            one_off_sequence = sequence[:i] + replacement + sequence[i+1:]
            one_off_sequences.append(one_off_sequence)
    return one_off_sequences

def map_error_corrections(barcodes):
    error_corrections = dict()
    for barcode in barcodes:
        error_corrections[barcode] = barcode
        one_off_BCs = find_one_off_errors(barcode)
        for one_off_BC in one_off_BCs:
            error_corrections[one_off_BC] = barcode
    return error_corrections

def map_error_corrections2(scar):
    error_corrections = dict()
    error_corrections[scar] = scar
    one_off_scars = find_one_off_errors(scar)
    for one_off_scar in one_off_scars:
        error_corrections[one_off_scar] = scar
    return error_corrections


def pairwise_levenshtein_distances(string, vector_of_strings):
    """
    Calculate pairwise levenshtein distances between a string and a vector of strings.
    """
    return [distance(string, seq) for seq in vector_of_strings]

"""get maps of a barcode that is one bp off to correct barcode"""
bc_1_map = map_error_corrections(BC_1s)
bc_2_map = map_error_corrections(BC_2s)
bc_3_map = map_error_corrections(BC_3s)
bc_4_map = map_error_corrections(BC_4s)


"""get maps of a sender that is one bp off to correct sender sequence"""
SR_map = map_error_corrections(SRs)

#the constant regions within spatial barcode
"""get maps of the ligation overhangs that is one bp off to correct overhang"""
# scars_one = map_error_corrections(find_one_off_errors("AACC"))
# scars_two = map_error_corrections(find_one_off_errors("ACAG"))

# scars_two = map_error_corrections(find_one_off_errors("ACAG"))
# scars_three = map_error_corrections(find_one_off_errors("CCTA"))
# scars_four = map_error_corrections(find_one_off_errors("TTCGAG"))
scars_one = map_error_corrections2("AACC")
scars_two = map_error_corrections2("ACAG")


# scars = "AACCACAGCCTATTCGAG"

#only checks the first two ligation scars

def determine_extension(sequence, overhang): 
    # print(sequence)   
    try:
        scars = scars_one[sequence[10+overhang:14+overhang]] + scars_two[sequence[24+overhang:28+overhang]]
        extender = 0
    except KeyError:
        try:
            scars = scars_one[sequence[11+overhang:15+overhang]] + scars_two[sequence[25+overhang:29+overhang]]
            extender = 1
        except KeyError:
            try:
                scars = scars_one[sequence[12+overhang:16+overhang]] + scars_two[sequence[26+overhang:30+overhang]]
                extender = 2
            except KeyError:
                try:
                    scars = scars_one[sequence[13+overhang:17+overhang]] + scars_two[sequence[27+overhang:31+overhang]]
                    extender = 3
                except KeyError:
                    #return something if none
                    extender = None
    return extender

def get_barcodes(sequence, bc_1_map, bc_2_map, bc_3_map, bc_4_map, SR_map, overhang):
    #figure out where errors are
        
    #assuming extension is zero
    interval_1 = (0+overhang, 10+overhang)
    interval_2 = (14+overhang, 24+overhang)
    interval_3 = (28+overhang, 38+overhang)
    interval_4 = (42+overhang, 52+overhang)
    UMI_interval = (70+overhang, 76+overhang)
    SR_interval = (76+overhang, 98+overhang)

    
    extension = determine_extension(sequence, overhang)
    #if none , still return whole sequence, write to sep file,
    if type(extension) is int:
        # Extract sequences for each barcode, UMI, and SR
        bc_1 = sequence[interval_1[0] + extension:interval_1[1] + extension]
        bc_2 = sequence[interval_2[0] + extension:interval_2[1] + extension]
        bc_3 = sequence[interval_3[0] + extension:interval_3[1] + extension]
        bc_4 = sequence[interval_4[0] + extension:interval_4[1] + extension]
        UMI = sequence[UMI_interval[0] + extension:UMI_interval[1] + extension]
        SR = sequence[SR_interval[0] + extension:SR_interval[1] + extension]
        
        try:
            SR = SR_map[SR[:-2]]  # Attempt to map SR assuming it is on the sense strand
        except KeyError:
            distances = pairwise_levenshtein_distances(SR, SRs)  # Calculate Levenshtein distances
            if distances:
                min_distance = min(distances)
                if min_distance < 5:
                    min_index = distances.index(min_distance)
                    SR = 'lev_' + SRs[min_index]  # Update SR to the closest match if within distance threshold
                else:
                    matches = [sr for sr in SRs if sr in sequence]
                    if matches:
                        SR = 'grep_' + matches[0]  # Use the first direct match from sequence
                    else:
                        SR = 'unmap_'+SR  # Mark as 'unmapped' if no matches found
            else:
                SR = 'unmap_'+SR  # Use 'unmapped' if no distances were calculated
        
        exact_match_or_not = [1, 1, 1, 1]
        try:
            check_index = BC_1s.index(bc_1)
        except ValueError:
            exact_match_or_not[0] = 0
        try:
            check_index = BC_2s.index(bc_2)
        except ValueError:
            exact_match_or_not[1] = 0
        try:
            check_index = BC_3s.index(bc_3)
        except ValueError:
            exact_match_or_not[2] = 0
        try:
            check_index = BC_4s.index(bc_4)
        except ValueError:
            exact_match_or_not[3] = 0
        if sum(exact_match_or_not) <= 2:
            return ['2', sequence, '', '', '']
        elif sum(exact_match_or_not) == 4:
            corrected_bc = [bc_1 + bc_2 + bc_3 + bc_4]
            return corrected_bc + [UMI, SR]
        else:
            if exact_match_or_not[0] == 0:
                try:
                    bc_1 = bc_1_map[bc_1]
                    corrected_bc = [bc_1 + bc_2 + bc_3 + bc_4]
                    return corrected_bc + [UMI, SR]
                except KeyError:
                    return ['2', sequence, '', '', '']
            if exact_match_or_not[1] == 0:
                try:
                    bc_2 = bc_2_map[bc_2]
                    corrected_bc = [bc_1 + bc_2 + bc_3 + bc_4]
                    return corrected_bc + [UMI, SR]
                except KeyError:
                    return ['2', sequence, '', '', '']
            if exact_match_or_not[2] == 0:
                try:
                    bc_3 = bc_3_map[bc_3]
                    corrected_bc = [bc_1 + bc_2 + bc_3 + bc_4]
                    return corrected_bc + [UMI, SR]
                except KeyError:
                    return ['2', sequence, '', '', '']
            if exact_match_or_not[3] == 0:
                try:
                    bc_4 = bc_4_map[bc_4]
                    corrected_bc = [bc_1 + bc_2 + bc_3 + bc_4]
                    return corrected_bc + [UMI, SR]
                except KeyError:
                    return ['2', sequence, '', '', '']


                    

            # bc_1 = bc_1_map[bc_1]
            # bc_2 = bc_2_map[bc_2]
            # bc_3 = bc_3_map[bc_3]
            # bc_4 = bc_4_map[bc_4]
            
            # corrected_bc = [bc_1 + bc_2 + bc_3 + bc_4]
            
            # return corrected_bc + [UMI, SR]
        
        # except KeyError: #will have KeyError if observed barcode not in mapping dictionary
        #     return ['2', sequence, '', '', '']#if line in df has 2, bc was not able to be error corrected
    else:
      return ['1', sequence, '', '', ''] #if line in df has 1, scars did not align

def get_sequences_into_df(R1_filename, R2_filename, bc_1_map, bc_2_map, bc_3_map, bc_4_map, SR_map, output_filename):
    """writing into txt file, then loading as dataframe"""

    output_f = open(output_filename, "w")
    output_f.write("R1_full_bc\tR1_UMI\tSR\tR2_UMI\tR2_full_bc\tSR_correct\n")
    with gzip.open(R1_filename, "rt") as f1, gzip.open(R2_filename, "rt") as f2:
        counter = 0
        for line_number, (line1, line2) in enumerate(zip(f1, f2)):
            # print(line1)
            # print(line2)
            # Print every second line out of 4
            if line_number % 4 == 1:
                seq1 = line1.rstrip()
                seq2 = line2.rstrip()
                seq1_data = get_barcodes(seq1, bc_1_map, bc_2_map, bc_3_map, bc_4_map, SR_map, overhang= 4)
                #print(seq1_data)
                seq2_data = get_barcodes(seq2, bc_1_map, bc_2_map, bc_3_map, bc_4_map, SR_map, overhang= 4)
                #print(seq2_data)
                correct = ''
                SR = ''
                if seq1_data[0] == '1' or seq2_data[0] =='1':
                    #type 1 error, scars do not align, 'junk' read
                    output_f.write(f"{'1'}\t{''}\t{''}\t{''}\t{''}\t{''}\n")
                elif seq1_data[0] == '2' or seq2_data[0] =='2':
                    #type 2 error, sender barcode doesnt match
                    # print(str(line_number)+"no")
                    output_f.write(f"{'2'}\t{''}\t{''}\t{''}\t{''}\t{''}\n")
                elif seq1_data[2].startswith('unmap_') and not seq2_data[2].startswith('unmap_'):
                    # print(str(line_number)+"yes")
                    if seq2_data[2].startswith('lev'):
                        correct = 'lev_r2'
                        SR = seq2_data[2][4:]
                    else:
                        correct = 'r2'
                        SR = seq2_data[2]
                    #read1 sender receiver sequence doesnt map, but read 2 does
                    output_f.write(f"{seq1_data[0]}\t{seq1_data[1]}\t{SR}\t{seq2_data[1]}\t{seq2_data[0]}\t{correct}\n")
                elif not seq1_data[2].startswith('unmap_') and seq2_data[2].startswith('unmap_'):
                    # print(str(line_number)+"yes")
                    if seq1_data[2].startswith('lev'):
                        correct = 'lev_r1'
                        SR = seq1_data[2][4:]
                    else:
                        correct = 'r1'
                        SR = seq1_data[2]
                    #read2 sender receiver sequence doesnt map, but read 1 does
                    output_f.write(f"{seq1_data[0]}\t{seq1_data[1]}\t{SR}\t{seq2_data[1]}\t{seq2_data[0]}\t{correct}\n")
                elif not seq1_data[2].startswith('unmap_') and not seq2_data[2].startswith('unmap_'):
                    # print(str(line_number)+"yes")
                    if seq1_data[2].startswith('lev') and not seq2_data[2].startswith('lev'):
                        correct = 'r2'
                        SR = seq2_data[2]
                    elif seq2_data[2].startswith('lev') and not seq1_data[2].startswith('lev'):
                        correct = 'r1'
                        SR = seq1_data[2]
                    else:
                        correct = 'lev_both'
                        SR = seq1_data[2][4:]
                    #both sequences map, just take read1
                    output_f.write(f"{seq1_data[0]}\t{seq1_data[1]}\t{SR}\t{seq2_data[1]}\t{seq2_data[0]}\t{correct}\n")
                else:
                    #neither map, just take r1?
                    output_f.write(f"{seq1_data[0]}\t{seq1_data[1]}\t{seq1_data[2]}\t{seq2_data[1]}\t{seq2_data[0]}\t{'neither'}\n")

            if counter % 1000000 == 0:
                print(f"Counter: {counter}")
            
            counter += 1

    print(f"# of reads in fastq: {counter/4}")

#output_f.write

#fastq file to process barcodes
R1_filename = sys.argv[1]
R2_filename = sys.argv[2]

# overhang = sys.argv[3]
output_filename = sys.argv[3]

#call function to write text file for loading dataframe
get_sequences_into_df(R1_filename,R2_filename, bc_1_map, bc_2_map, bc_3_map, bc_4_map, SR_map,output_filename)


### this part is for dropping NAs from text file
#load df from text file
single_lib = pd.read_csv(sys.argv[3], sep="\t")

scars_unaligned_count = len(single_lib[single_lib["R1_full_bc"] == "1"])
bc_uncorrected = len(single_lib[single_lib["R1_full_bc"] == "2"])
sender_uncorrected = len(single_lib[single_lib["SR_correct"] == "neither"])
sender_r1_corrected = len(single_lib[single_lib["SR_correct"] == "r1"])
sender_r2_corrected = len(single_lib[single_lib["SR_correct"] == "r2"])
sender_lev_r1_corrected = len(single_lib[single_lib["SR_correct"] == "lev_r1"])
sender_lev_r2_corrected = len(single_lib[single_lib["SR_correct"] == "lev_r2"])
sender_lev_both_corrected = len(single_lib[single_lib["SR_correct"] == "lev_both"])


print(f"Unaligned: {scars_unaligned_count}")
print(f"Uncorrected bc: {bc_uncorrected}")
print(f"Corrected sender r1: {sender_r1_corrected}")
print(f"Corrected sender r2: {sender_r2_corrected}")
print(f"Corrected sender levenshtein r1: {sender_lev_r1_corrected}")
print(f"Corrected sender levenshtein r1: {sender_lev_r2_corrected}")
print(f"Corrected sender levenshtein both: {sender_lev_both_corrected}")

single_lib.replace("", float("NaN"), inplace=True)
single_lib.replace("1", float("NaN"), inplace=True)
single_lib.replace("2", float("NaN"), inplace=True)

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
sys.argv[3]: write dataframe line by line to csv file (has a lot of zeros), has to be .txt file
sys.argv[4]: after dropping NAs from dataframe, write final lib into csv file

Now including SR correction terms 
term     //	 meaning
both     //  both r1 and r2 did not need error correction, r1 seq will be reported
r1	     //  r2 was not mappable/corrected, r1 used
r2	     //  r1 was not mappable/corrected, r2 used
lev_both //	 both r1 and r2 were corrected using levenshtein distances, r1 reported
lev_r1	 //  r2 was not mappable/corrected, r1 was corrected using levenshtein
lev_r2	 //  r1 was not mappable/corrected, r2 was corrected using levenshtein
neither	 //  both r1 and r2 were not mappable/corrected
"""
