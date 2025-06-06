"""
This module contains functions for processing and analyzing CRISPR array data.
It includes functions for finding Cas proteins, and ensuring they are within
the flanking regions of CRISPR arrays.
"""

import re
import os
import argparse
import subprocess
import glob
import logging
from Bio import SeqIO
import pandas as pd

def find_equal_positions(line):
    """
    Find continuous blocks of the = character in a string and return their start and end
    positions as a list of tuples.
    """
    positions = []
    start = -1  # Initialize start position
    for i, char in enumerate(line):
        if char == '=':
            if start == -1:  # Starting a new block
                start = i
        else:
            if start != -1:  # Ending the current block
                positions.append((start, i - 1))
                start = -1  # Reset start position

    # If the string ends with '='
    if start != -1:
        positions.append((start, len(line) - 1))
    return positions

def extract_crispr_positions_pilercr(file_path):
    """
    Extract pilercr positions
    """
    with open(file_path, 'r', encoding = 'utf-8') as file:
        crispr_positions = []
        sequence_id = None
        processing = False
        for line in file:
            # Start processing when "SUMMARY BY POSITION" is found
            if "SUMMARY BY POSITION" in line:
                processing = True
                continue
            # Process only after "SUMMARY BY POSITION"
            if processing:
                # strip will remove white spacers at the beginning, this affects length difference
                line = line.rstrip()
                # Check for a new sequence (lines starting with '>')
                if line.startswith('>'):
                    # Extract the sequence ID (until the first white space)
                    sequence_id = line.split()[0][1:]

                elif line.startswith('='):  # update code to deal with various seq id
                    columns = find_equal_positions(line)

                # Process CRISPR array lines (they have the format of the columns)
                elif line != "" and not line.startswith('=') and not line.startswith('Array'):
                    extracted_data = [line[start:end + 1].strip() for start, end in columns]

                    # Ensure we have enough columns to process
                    if len(extracted_data) >= 4:
                        position = int(extracted_data[2])  # Position column
                        length = int(extracted_data[3])  # Length column

                        # Calculate the start and end positions
                        start = position
                        end = position + length - 1

                        # Store the result in the list
                        crispr_positions.append([f"{sequence_id}", int(start), int(end)])

        return crispr_positions

def extract_crispr_positions_minced(file_path):
    """
    Extract minced crispr positions
    """
    if os.path.getsize(file_path) > 0:
        # Regular expression patterns
        # Matches the content between quotes after 'Sequence'
        sequence_pattern = r"Sequence\s+'([^']+)'"
        # Matches the CRISPR range line
        range_pattern = r"CRISPR\s+\d+\s+Range:\s+(\d+)\s*-\s*(\d+)"

        # Initialize variables
        crispr_positions = []
        current_sequence_id = None

        # Open and read the file line by line
        with open(file_path, 'r', encoding = 'utf-8') as file:
            for line in file:
                # Check if the line contains a sequence ID
                if "Sequence" in line:
                    sequence_match = re.search(sequence_pattern, line)
                    current_sequence_id = sequence_match.group(1) # Capture the current sequence ID

                # Check if the line contains a CRISPR range
                if "Range" in line and current_sequence_id:
                    range_match = re.search(range_pattern, line)
                    start, end = range_match.groups()
                    crispr_positions.append([f"{current_sequence_id}", int(start), int(end)])

        return crispr_positions
    return []

def rename_records(records, name_sep = "ENA|"):
    """
    Processes a list of records and appends a unique identifier to each record's name based on
    occurrences of an identifier extracted from the record string. Records with the same
    identifier are uniquely renamed by appending a number (INDEX1, INDEX2, etc.).
    """
    # Dictionary to track occurrences of unique identifiers (part after first "|" and before first ",")
    id_counts = {}
    renamed_records = []
    for record, start, end in records:
        identifier = record.split(name_sep)[-1]  # record.split('ENA|')[-1]

        # Check if this identifier already exists in id_counts
        if identifier not in id_counts:
            id_counts[identifier] = 1
        else:
            id_counts[identifier] += 1

        # Add suffix 1, 2, etc., based on count
        new_first_part = f"{record}INDEX{id_counts[identifier]}"

        # Construct the renamed record and add to the result list
        renamed_records.append([f"{new_first_part}", start, end])
    return renamed_records

def group_records_by_id(records):
    """
    Groups records by the base crispr_id in a dictionary.
    """
    grouped_records = {}
    for record in records:
        # Extract the base crispr_id by splitting on the underscore (_MINCED or _PILERCR)
        base_id = record[0]
        if base_id not in grouped_records:
            grouped_records[base_id] = [record]
        else:
            grouped_records[base_id].append(record)
    return grouped_records

def merge_crispr_records(minced_records, pilercr_records,
                         position_diff_threshold = 1000, name_sep = "ENA|"):
    """
    Merge CRISPR records from minced and pilercr based on position differences.
    since later we want the seq 20k up and down the crispr array. so if position doesn't
    very a lot, it can be treated as the same ?
    """
    merged_crispr_records = minced_records.copy()

    # Group records by crispr_id
    minced_by_id = group_records_by_id(minced_records)

    for pilercr_record in pilercr_records:
        crispr_id, pilercr_start, pilercr_end = pilercr_record
        match_found = False
        if crispr_id in minced_by_id:
            # Compare positions if the crispr_id exists in minced
            minced_records = minced_by_id[crispr_id]
            for minced_record in minced_records:
                minced_start, minced_end = minced_record[1], minced_record[2]
                # Check if the positions are within the threshold
                if abs(pilercr_start - minced_start) <= position_diff_threshold and abs(
                        pilercr_end - minced_end) <= position_diff_threshold:
                    match_found = True  # Mark the match and break the loop
                    break  # Skip, treat as the same record
        # If it's not the same record, add it to the merged list
        if not match_found:
            merged_crispr_records.append(["PILERCR_" + crispr_id, pilercr_start, pilercr_end])
    # rename records，so easier to identify crispr contig, if one contig
    # has multiple crispr array regions
    merged_crispr_records = rename_records(merged_crispr_records, name_sep = name_sep)
    return merged_crispr_records

def write_merged_records(filename, records):
    """
    Writes the merged CRISPR records to a text file.
    """
    with open(filename, 'w', encoding = 'utf-8') as file:
        for record in records:
            file.write(f"{record[0]},{record[1]},{record[2]}\n")

def to_dict_remove_duplicates(sequences):
    """
    Removes duplicates from the sequences (ensure that only one record with a given id is kept).
    Takes a collection of sequences and returns a dictionary where the keys are the unique id
    attributes of the record objects and the values are the corresponding record objects
    themselves.
    """
    return {record.id: record for record in sequences}

def write_merged_seqs(unzipped_fasta_file, flanking_size_filename, merged_fasta_file, merged_crispr_records,
                      flanking_size = 20000):  # 20000
    """
    Writes the merged CRISPR record seqs to a fasta format file.
    """
    try:
        record_dict = SeqIO.to_dict(SeqIO.parse(unzipped_fasta_file, "fasta"))
    # sometimes the input seqs contain duplicated records (e.g. GCA_000157055.1)
    except ValueError as e:
        if "Duplicate key" in str(e):
            record_dict = to_dict_remove_duplicates(SeqIO.parse(unzipped_fasta_file, "fasta"))
        else:
            raise  # raise any other ValueErrors

    with open(merged_fasta_file, 'w', encoding = 'utf-8') as file, open(flanking_size_filename, 'w', encoding = 'utf-8') as flanking_file:
        for seq_id, start_pos, end_pos in merged_crispr_records:
            # Regular expression to match text after "PILERCR_"
            pure_seq_id = seq_id.split("PILERCR_")[-1]
            # back from rename id to pure id, "_" to "INDEX"
            pure_seq_id = pure_seq_id.split("INDEX")[0]
            contig_seq = record_dict[pure_seq_id].seq
            contig_len = len(contig_seq)
            if contig_len > 5000:
                start_pos = max(1, start_pos - flanking_size)
                end_pos = min(contig_len, end_pos + flanking_size)
                record_seq = contig_seq[start_pos - 1:end_pos]
                # since the crispr cas seq depends on where the array is found and then extend it
                # up/down stream, so use seq_id with array found method and if repeat with suffix
                file.write(f">{seq_id}\n{record_seq}\n")

                actual_flanking_size_left = start_pos - max(1, start_pos - flanking_size)
                actual_flanking_size_right = min(contig_len, end_pos + flanking_size) - end_pos
                flanking_file.write(f"{seq_id}\t{actual_flanking_size_left}\t{actual_flanking_size_right}\n")

def run_prodigal_gv_on_crispr_array_contigs():
    """
    Run prodigal-gv to perform gene prediction and check if the resulting file
    contains something.
    """
    command = ["prodigal-gv", "-p", "meta", "-i", f"{merged_fasta}", "-a", f"{protein_fasta}", "-m"]
    # Run prodigal, suppressing output by redirecting to /dev/null
    with open('/dev/null', 'w', encoding = 'utf-8') as devnull:
        subprocess.run(command, stdout = devnull, stderr = subprocess.STDOUT, check = False)
    if os.path.getsize(f"{protein_fasta}") == 0: # check if is file empty
        return False
    return True

def check_and_empty_if_only_comments(file_path):
    """
    Checks whether a given file consists only of comment lines (lines starting with #).
    If the file only contains comments, it deletes the file. If the file contains any
    non-comment lines, it returns False and does nothing to the file.
    """
    with open(file_path, 'r', encoding = 'utf-8') as file:
        for line in file:
            # Check if any line doesn't start with "#"
            if not line.strip().startswith("#"):
                return False  # Exit if a non-comment line is found
    os.remove(file_path)  # delete if only comments
    return True

def run_hmmsearch_to_find_cas_proteins(hmm_files_folder):
    """
    Run hmmsearch
    """
    cas_finding_results_sub_folder = f"{output_folder}"
    # Create the directory if it doesn't exist
    os.makedirs(cas_finding_results_sub_folder, exist_ok = True)
    # Define the input paths
    protein_file = f"{protein_fasta}"
    # Get all the .hmm files
    hmm_files = glob.glob(f'{hmm_files_folder}/*.hmm')

    # Loop over each hmm file
    for hmm_file in hmm_files:
        # Construct the output file path
        output_file = os.path.join(cas_finding_results_sub_folder,
                       f"{os.path.basename(hmm_file).replace('.hmm', '')}_hmmer_tb_table.txt")
        # run hmmsearch
        command = [
            "hmmsearch", "--noali", "-o", "/dev/null",  # Redirect main output to /dev/null
            "--tblout", output_file, hmm_file, protein_file,
        ]
        subprocess.run(command, check = True)
        check_and_empty_if_only_comments(output_file)

def read_protein_coding_records_of_crispr_array_containing_contigs():
    """
    Reads a FASTA file containing protein-coding records, processes the header information of
    each sequence, and creates a Pandas DataFrame, with columns for the CRISPR contig ID, protein
    ID, leftmost and rightmost coordinates, and strand direction.
    """
    file_path = f"{protein_fasta}"
    result_list = []
    with open(file_path, 'r', encoding = 'utf-8') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace from the entire line

            if line.startswith(">"):  # Check if the line starts with '>'
                # Split the line on spaces to extract the first word (ID part)
                parts = line[1:].split("#")
                parts = [part.strip() for part in parts]
                protein_id, leftmost_coord, rightmost_coord, strand = parts[0:4]
                crispr_contig_id = "_".join(protein_id.split("_")[:-1])
                result_list.append((crispr_contig_id, protein_id, leftmost_coord, rightmost_coord,
                                    strand))

    # Convert to DataFrame
    df = pd.DataFrame(result_list, columns = ['crispr_contig_id', 'protein_id', 'leftmost_coord',
                               'rightmost_coord', 'strand'])
    # Convert the coordinate and strand columns to integer types
    df['leftmost_coord'] = df['leftmost_coord'].astype(int)
    df['rightmost_coord'] = df['rightmost_coord'].astype(int)
    df['strand'] = df['strand'].astype(int)
    return df

def get_existing_cas_result_files(data_folder, suffix):
    """
    Find and process files in data_folder that match a given suffix
    """
    all_file_names = glob.glob(f"{data_folder}*{suffix}", )
    all_file_names = [os.path.basename(l) for l in all_file_names]
    all_file_ids = [l.split(f"{suffix}")[0] for l in all_file_names]
    return set(all_file_ids)

def get_cas_find_results_per_hmm(hmm_id, cas_finding_results_sub_folder):
    """
    Process HMMER results for a given HMM (Hidden Markov Model) ID. It reads the file,
    extracts relevant data from each line (ignoring comment lines), and organizes
    the data into a list of lists. Each list contains information related to a protein,
    its associated CRISPR contig, and evaluation scores from the HMMER output.
    """
    cas_finding_list = []
    file_path = f"{cas_finding_results_sub_folder}{hmm_id}_hmmer_tb_table.txt"
    with open(file_path, 'r', encoding = 'utf-8') as file:
        for line in file:
            # Ignore comment lines starting with "#"
            if not line.startswith("#"):
                # Split the line by spaces (this will handle multiple spaces between columns)
                fields = line.strip().split()
                fields = [field.strip() for field in fields]
                protein_id, full_seq_eval, full_seq_score = fields[0], fields[4], fields[5]
                crispr_contig_id = "_".join(protein_id.split("_")[:-1])
                cas_finding_list.append([crispr_contig_id, hmm_id, protein_id,
                                         full_seq_eval, full_seq_score])
    return cas_finding_list

def get_cas_find_results_per_sample():
    """
    Collect and process HMMER results for multiple samples, convert the data into a pandas
    DataFrame, and return the results.
    """
    cas_finding_results_sub_folder = f"{output_folder}/"
    cas_finding_result_hmm_ids = get_existing_cas_result_files(cas_finding_results_sub_folder,
                                                             "_hmmer_tb_table.txt")
    cas_finding_list = []
    for hmm_id in cas_finding_result_hmm_ids:
        cas_finding_list.extend(get_cas_find_results_per_hmm(hmm_id,
                                                             cas_finding_results_sub_folder))
    df = pd.DataFrame(cas_finding_list, columns = ['crispr_contig_id', 'hmm_id', 'protein_id',
                                                 'full_seq_eval', 'full_seq_score'])
    # Convert the coordinate and strand columns to integer types
    df['full_seq_eval'] = df['full_seq_eval'].astype(float)
    df['full_seq_score'] = df['full_seq_score'].astype(float)
    return df

def check_coordinates(row, crispr_contig_record_dict):
    """
    Check if the coordinates of a CRISPR array, found in a row of data, are within the valid
    range of the corresponding CRISPR contig. It performs two main checks: verifying that the
    coordinates do not exceed the length of the contig, and ensuring that the coordinates are
    either close to the beginning or end of the contig.
    """
    crispr_contig_len = crispr_contig_record_dict.get(row['crispr_contig_id'])
    leftmost_coord, rightmost_coord = row['leftmost_coord'], row['rightmost_coord']
    try:
        assert leftmost_coord <= crispr_contig_len
        assert rightmost_coord <= crispr_contig_len
    except:
        logging.info(f"{sample}'s crispr_array found Cas beyond contig region: "
                     f"{crispr_contig_len}; {row};")
        raise

    if (rightmost_coord < 20000) or (leftmost_coord > crispr_contig_len - 20000):
        return True, row.values.tolist()
    return False, None

def read_crispr_array_containing_contigs(return_len_only = False):
    """
    Read a FASTA file containing CRISPR array contigs, process them using the SeqIO module
    from Biopython, and return either the contig records themselves or just the lengths of
    those contigs.
    """
    crispr_contig_record_dict = SeqIO.to_dict(SeqIO.parse(f"{merged_fasta}", "fasta"))
    if return_len_only:
        crispr_contig_record_dict = {k:len(v) for k, v in crispr_contig_record_dict.items()}
    return crispr_contig_record_dict

def filtering_cas_results_in_flanking_region(protein_finding_join_cas_finding):
    """
    Filter rows in a DataFrame that contain CRISPR array-related information, based
    on whether the coordinates of the Cas genes are within a specified
    "flanking region" of the contig. It performs a series of checks using external helper
    functions like check_coordinates() to validate the positions of the genes are within
    a valid range of the contig.
    """
    crispr_contig_record_dict = read_crispr_array_containing_contigs(return_len_only = True)
    valid_rows = []
    cross_flanking_check = []
    # Iterate over each row in the DataFrame
    for index, row in protein_finding_join_cas_finding.iterrows():
        result = check_coordinates(row, crispr_contig_record_dict)
        cross_flanking_check.append(result[0])
        # If the result is not None, append the row to valid_rows
        if result[1] is not None:
            valid_rows.append(result[1])
    if not all(cross_flanking_check):
        logging.info(f"{sample}'s crispr_array found Cas not in flanking region")

    # Convert the list of valid rows back to a DataFrame
    protein_finding_join_cas_finding_flanking_region = pd.DataFrame(valid_rows,
                                   columns = protein_finding_join_cas_finding.columns)
    return protein_finding_join_cas_finding_flanking_region

def get_crispr_operon_records():
    """
    Process and filter CRISPR operon-related data. It merges protein-coding records
    for CRISPR arrays and Cas protein findings, and removes rows where hmm_id is
    NA and checks whether the found cas proteins are in the flanking region.
    """
    protein_coding_records_df = read_protein_coding_records_of_crispr_array_containing_contigs()
    cas_finding_records_df = get_cas_find_results_per_sample()
    protein_finding_join_cas_finding = pd.merge(protein_coding_records_df,
                                              cas_finding_records_df,
                                              on=["crispr_contig_id", "protein_id"], how="outer")
    protein_finding_join_cas_finding = (
        protein_finding_join_cas_finding)[~ protein_finding_join_cas_finding.hmm_id.isna()]

    # filter results by checking where the found cas protein are in the flanking region
    protein_finding_join_cas_finding_flanking_region = (
        filtering_cas_results_in_flanking_region(protein_finding_join_cas_finding))
    return protein_finding_join_cas_finding_flanking_region

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify crispr containing contigs.")
    parser.add_argument('--minced_file', action="store", dest="minced_file")
    parser.add_argument('--pilercr_file', action="store", dest="pilercr_file")
    parser.add_argument('--merged_file', action="store", dest="merged_file")
    parser.add_argument('--merged_fasta', action="store", dest="merged_fasta")
    parser.add_argument('--unzipped_fasta', action="store", dest="unzipped_fasta")
    parser.add_argument('--protein_fasta', action="store", dest="protein_fasta")
    parser.add_argument('--hmm_files_path', action="store", dest="hmm_files_path")
    parser.add_argument('--output_folder', action="store", dest="output_folder")
    parser.add_argument('--sample', action="store", dest="sample")
    parser.add_argument('--cas_flanking_region', action="store", dest="cas_flanking_region")
    parser.add_argument('--flanking_size_filename', action="store", dest="flanking_size_filename")
    args = parser.parse_args()

    minced_file = args.minced_file
    pilercr_file = args.pilercr_file
    merged_file = args.merged_file
    merged_fasta = args.merged_fasta
    unzipped_fasta = args.unzipped_fasta
    protein_fasta = args.protein_fasta
    hmm_files_path = args.hmm_files_path
    output_folder = args.output_folder
    sample = args.sample
    cas_flanking_region = args.cas_flanking_region
    flanking_size_filename = args.flanking_size_filename

    # Extract minced positions
    minced_positions = extract_crispr_positions_minced(minced_file)

    # Extract pilercr output
    pilercr_positions = extract_crispr_positions_pilercr(pilercr_file)

    # Merging the records
    merged_records = merge_crispr_records(minced_positions, pilercr_positions, name_sep="ENA|")

    # Writing the merged records to an output file
    write_merged_records(merged_file, merged_records)
    write_merged_seqs(unzipped_fasta, flanking_size_filename, merged_fasta, merged_records)

    # Get protein coding region
    protein_coding_status = run_prodigal_gv_on_crispr_array_contigs()

    if protein_coding_status:
        # finding cas position by searching via hmm profile
        run_hmmsearch_to_find_cas_proteins(hmm_files_path)

        # locate crispr array, cas position in the contigs , i.e., get cas operons
        # and filtering by check if they are in the flanking region
        protein_finding_join_cas_finding_flankingregion = get_crispr_operon_records()
        protein_finding_join_cas_finding_flankingregion.to_csv(f"{cas_flanking_region}",
                                                               header = True, index = False,
                                                               sep = "\t")

    else:
        logging.info(f"{sample} crispr contig has no protein coding region")
