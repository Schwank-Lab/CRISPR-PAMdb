import os
import subprocess
import glob
import shutil
from multiprocessing import Pool

import pandas as pd
from Bio import SeqIO
import logging
import argparse
import pathlib
from crispr_containing_contigs import find_equal_positions

# in rare case DNA of Cas contain "N", this will be translated to X in the CDS
# >ANAN16-1_SAMN03842439-S001_MAG_00000505-scaffold_47INDEX1_3

# write all cas protein to one file , later to be used for clustering
########################################################
def return_seq_from_crispr_contig(merged_fasta, crispr_contig_id, start_pos, end_pos, strand, tsv_file):
    """
    Retrieves a protein sequence from a specified region of a contig in a CRISPR array dataset.
    """
    try:
        crispr_contig_seq_current = SeqIO.to_dict(SeqIO.parse(merged_fasta, "fasta"))[crispr_contig_id]
        return_seq = crispr_contig_seq_current[start_pos - 1:end_pos].seq
        if strand == -1:
            return_seq = "".join([ATCG_mapping[s] for s in return_seq])  # this will get seq on the opposite strand from left to right
            return_seq = return_seq[::-1]  # this will get seq on the opposite strand from right to left , same as results from prodigal
        return return_seq
    except Exception as e:
        print(f"An error occurred: {e};{tsv_file};{crispr_contig_id}")
        raise

def get_existing_file(all_file_names, return_only_non_empty_file = False):
    """
    Retrieves the identifiers of files in a specified folder with a given suffix (file extension).
    """
    if return_only_non_empty_file:
        logging.info(f"len before excluding empty files:{len(all_file_names)}")
        all_file_names = [f for f in all_file_names if os.path.getsize(f) > 0] # # File is not empty
    all_file_names = [os.path.basename(l) for l in all_file_names]
    all_file_ids = [l.split(f".tsv")[0] for l in all_file_names]
    return set(all_file_ids)

def create_folder_if_not_existing(folder_path):
    """
    Create folder if it does not exist
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def get_cas_seq_from_crispr_core_operon_pos(tsv_file, crispr_core_operon_pos, cas_type, merged_fasta):
    """
    Extracts the sequence list of Cas proteins (associated with a specific cas_type) from a given
    set of CRISPR operon positions for a dataset.
    """
    try:
        grouped = crispr_core_operon_pos.groupby("crispr_contig_id")
        cas_protein_seq_list = list()
        for crispr_contig_id, group in grouped:  # process per crispr_contig
            cas_row = group[group['hmm_id'].str.contains(cas_type, case = False, na = False)]
            cas_row_list = cas_row.iloc[0].tolist()
            crispr_contig_id, protein_id, leftmost_coord, rightmost_coord, strand = cas_row_list[0], cas_row_list[1], int(
                cas_row_list[2]), int(cas_row_list[3]), cas_row_list[4]
            cas_protein_seq = return_seq_from_crispr_contig(merged_fasta, crispr_contig_id, leftmost_coord, rightmost_coord,
                                                            strand, tsv_file)  # also here better use protein seq?, as different codon might for same codon ?
            cas_protein_seq_list.append(("#".join((tsv_file, protein_id)), cas_protein_seq))
        return cas_protein_seq_list
    except IndexError as e:
        raise e

def append_to_fasta(sequences, file_name, write_mode = "a"):
    """
    Appends or writes a list of sequences to a FASTA file.
    """
    with open(file_name, write_mode) as fasta_file:
        for seq_id, seq in sequences:
            fasta_file.write(f">{seq_id}\n")  # Write the sequence ID as a header
            fasta_file.write(f"{seq}\n")  # Write the sequence

def collect_all_cas_in_one_file(crispr_operon_all_pos_in_crispr_contig_files, all_cas_proteins_folder, cas_type, merged_fastas):
    """
    Consolidates specific Cas protein sequences from multiple input files into a single FASTA file.
    """
    crispr_operon_all_pos_in_crispr_contig_files = sorted(list(crispr_operon_all_pos_in_crispr_contig_files))

    file_name = f"{all_cas_proteins_folder}all_{cas_type}.fasta"
    os.remove(file_name) if os.path.exists(file_name) else None  # delete file if exists so avoid repeatedly appending seqs later

    for i in range(len(crispr_operon_all_pos_in_crispr_contig_files)):
        if os.path.exists(crispr_operon_all_pos_in_crispr_contig_files[i] + "2"):
            try:
                crispr_operon_all_pos_in_crispr_contig_result = pd.read_csv(crispr_operon_all_pos_in_crispr_contig_files[i] + "2", header = 0, index_col = None, sep ="\t")
                crispr_core_operon_pos = crispr_operon_all_pos_in_crispr_contig_result[~(
                            (crispr_operon_all_pos_in_crispr_contig_result["hmm_id"] == "0") & (
                                crispr_operon_all_pos_in_crispr_contig_result["protein_id"] != "repeat") & (
                                        crispr_operon_all_pos_in_crispr_contig_result["protein_id"] != "spacer"))]
                cas_seq = get_cas_seq_from_crispr_core_operon_pos(crispr_operon_all_pos_in_crispr_contig_files[i] + "2", crispr_core_operon_pos, cas_type, merged_fastas[i])  # further get cas_type cas proteins only,here should we actually use DNA seq of cas protein, not protein seq ?
                append_to_fasta(cas_seq, file_name)

            except IndexError as e:
                print(f"There were no {cas_type} hits for {crispr_operon_all_pos_in_crispr_contig_files[i]}2.")
                continue


def mmseqs_easy_clustering(all_cas_proteins_folder, cas_type, threads_num):
    """
    Cluster cas sequences using mmseqs.
    # "--min-seq-id", "0.98", "-c", "0.98", get 509 clustering
    # "--min-seq-id", "0.98", "-c", "0.95", get 507 clusters
    # "--min-seq-id", "0.98", "-c", "0.95", get 508 clusters
    # "--min-seq-id", "0.98", "-c", "1", get 524 clusters
    # "--min-seq-id", "0.99", "-c", "1",  get 562 clusters
    # after testing set as 0.98, PAMpredict also use 0.98
    """
    fasta_name = f"{all_cas_proteins_folder}all_{cas_type}.fasta"
    cluster_name = os.path.join(all_cas_proteins_folder, f"mmseqs2_clustering/{cas_type}_cluster")
    tmp_name = os.path.join(all_cas_proteins_folder, f"mmseqs2_clustering/tmp")
    command_output_dict = dict(stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    sub_cmd = ["mmseqs", "easy-cluster", fasta_name, cluster_name, tmp_name, "--min-seq-id", "0.98", "-c", "1",
               "--cov-mode", "0", "--cluster-mode", "0", "--threads", str(threads_num)]
    result = subprocess.run(sub_cmd, universal_newlines = True, **command_output_dict)  # **command_output_dict, add this to silicene output
    return result

def extract_crispr_spacer_repeat_positions_per_array_pilercr(file_path, array_start = 0):
    """
    Processes pilercr file containing CRISPR array information and extracts positions, repeat
    lengths, and spacer lengths for CRISPR arrays.
    """
    with open(file_path, 'r') as file:
        in_block = False
        spacer = 1000000000
        for line in file:
            # Check for the "SUMMARY BY SIMILARITY" line
            if "SUMMARY BY SIMILARITY" in line:
                break  # Stop processing the file when this line is found

            # Check for lines that start with "="
            if line.startswith("="):
                if not in_block:  # beginning of current block
                    columns = find_equal_positions(line)
                in_block = not in_block  # Toggle block state
                block_pos_repeat_spacer = list()
            elif in_block:
                # Collect lines that are part of a block
                extracted_data = [line[start:end + 1].strip() for start, end in columns]  #
                # get pos. repeat len, spacer length
                pos, repeat = int(extracted_data[0]), int(extracted_data[1])
                spacer = int(extracted_data[3]) if extracted_data[3] else None  # the None spacer after last repeat
                block_pos_repeat_spacer.append((pos, repeat, spacer))
            if spacer is None and block_pos_repeat_spacer and block_pos_repeat_spacer[0][0] == array_start:
                return block_pos_repeat_spacer


def extract_crispr_spacer_repeat_positions_per_array_minced(file_path, array_start = 0):
    """
    Processes minced file containing CRISPR array data to extract information about repeat
    positions, repeat lengths, and spacer lengths.
    """
    with open(file_path, 'r') as file:
        in_block = False
        spacer = 1000000000
        for line in file:
            line = line.strip()
            # Check for lines that start with "==="
            if line.startswith("--"):
                in_block = not in_block  # Toggle block state
                block_pos_repeat_spacer = list()
            elif in_block:
                # Collect lines that are part of a block
                extracted_data = [s.strip() for s in line.split()]
                if len(extracted_data) > 2:
                    pos, repeat, spacer = int(extracted_data[0]), int(extracted_data[4].split(",")[0]), int(
                        extracted_data[5])
                elif len(extracted_data) == 2:
                    pos, repeat = int(extracted_data[0]), len(extracted_data[1])
                    spacer = None
                block_pos_repeat_spacer.append((pos, repeat, spacer))
            if spacer is None and block_pos_repeat_spacer and block_pos_repeat_spacer[0][0] == array_start:
                return block_pos_repeat_spacer

def transfer_ori_pos_to_crispr_contig_pos(block_pos_repeat_spacer, array_start, flanking_size = 20000):
    """
    Converts positions from an original CRISPR array to positions relative to a CRISPR-containing contig,
    accounting for a flanking region around the array.
    """
    repeat_spacer_crispr_contig_pos = list()
    for pos, repeat, spacer in block_pos_repeat_spacer:
        repeat_start, repeat_end = flanking_size + 1 + pos - array_start, flanking_size + 1 + pos - array_start + repeat - 1
        if spacer is not None:
            spacer_start, spacer_end = flanking_size + 1 + pos - array_start + repeat, flanking_size + 1 + pos - array_start + repeat + spacer - 1 # here +1, to synchronised position from other files,
                            # it's start position in the genome, so start with 20 001,not flanking size 20 000
                            # -1, because the start position is already part of length of repeat,
        else:
            spacer_start, spacer_end = None, None
        repeat_spacer_crispr_contig_pos.append([(repeat_start, repeat_end), (spacer_start, spacer_end)])
    return repeat_spacer_crispr_contig_pos

def get_repeat_spacer_crispr_contig_pos_pilercr(crispr_contig_spos_in_ori, pilercr_output_file, flanking_size_left):
    """
    Processes the repeat and spacer positions of pilercr CRISPR arrays for a specific contig and returns
    their positions in a contig-relative coordinate system.
    """
    # notice this return position specific to crispr_contig_id, crispr_contig_spos_in_ori is unique to crispr_contig_id
    block_pos_repeat_spacer = extract_crispr_spacer_repeat_positions_per_array_pilercr(pilercr_output_file,
                                                                                       array_start = crispr_contig_spos_in_ori)
    repeat_spacer_crispr_contig_pos = transfer_ori_pos_to_crispr_contig_pos(block_pos_repeat_spacer,
                                                                            array_start = crispr_contig_spos_in_ori,
                                                                            flanking_size = flanking_size_left)
    return repeat_spacer_crispr_contig_pos


def get_repeat_spacer_crispr_contig_pos_minced(crispr_contig_spos_in_ori, minced_output_file, flanking_size_left):
    """
    Processes the repeat and spacer positions of minced CRISPR arrays for a specific contig and returns
    their positions in a contig-relative coordinate system.
    """
    # notice this return position specific to crispr_contig_id
    block_pos_repeat_spacer = extract_crispr_spacer_repeat_positions_per_array_minced(minced_output_file,
                                                                                      array_start = crispr_contig_spos_in_ori)
    repeat_spacer_crispr_contig_pos = transfer_ori_pos_to_crispr_contig_pos(block_pos_repeat_spacer,
                                                                            array_start = crispr_contig_spos_in_ori,
                                                                            flanking_size = flanking_size_left)
    return repeat_spacer_crispr_contig_pos


def merge_crispr_operon_all_pos_in_crispr_contig(crispr_operon_filtered_frame, merged_crispr_array_record,
                                                 flanking_size_record, protein_coding_records_df, pilercr_output_file, minced_output_file):
    """
    Integrates various CRISPR-related data sources for a given dataset, aligning and merging information
    about CRISPR arrays, Cas proteins, spacers, repeats, and other protein coding regions within a
    contig. It organizes and returns a unified data structure, sorted by genomic positions.
    """
    grouped = crispr_operon_filtered_frame.groupby("crispr_contig_id")
    crispr_operon_all_pos_in_crispr_contig_list_frame = list()
    for crispr_contig_id, group in grouped:  # process per crispr_contig
        crispr_contig_id, crispr_contig_spos_in_ori, crispr_contig_epos_in_ori = \
        merged_crispr_array_record.loc[merged_crispr_array_record[0] == crispr_contig_id].iloc[0]  # get merged crispr array record
        flanking_size_left = flanking_size_record.loc[flanking_size_record[0] == crispr_contig_id, 1].values[0]
        if "PILERCR" in crispr_contig_id:  # get position of repeat and spacer from Pilercr or minced and transform them to position in crispr_contig
            repeat_spacer_crispr_contig_pos = get_repeat_spacer_crispr_contig_pos_pilercr(crispr_contig_spos_in_ori,
                                                                                          pilercr_output_file, flanking_size_left)
        else:
            repeat_spacer_crispr_contig_pos = get_repeat_spacer_crispr_contig_pos_minced(crispr_contig_spos_in_ori,
                                                                                         minced_output_file, flanking_size_left)

        repeat_spacer_crispr_contig_pos_frame = pd.DataFrame(
            [(crispr_contig_id, "repeat", e[0], e[1]) if idx == 0 else ((crispr_contig_id, "spacer", e[0], e[1])) for l
             in repeat_spacer_crispr_contig_pos for idx, e in enumerate(l)],
            columns = ["crispr_contig_id", "protein_id", "leftmost_coord", "rightmost_coord"])
        repeat_spacer_crispr_contig_pos_frame = repeat_spacer_crispr_contig_pos_frame.dropna(subset = ['leftmost_coord'])
        crispr_operon_all_pos_in_crispr_contig = pd.concat([group, repeat_spacer_crispr_contig_pos_frame], axis = 0,
                                                         sort = False).fillna(
            0)  # save 0 not Nan so later write to file something could occupy file
        crispr_operon_all_pos_in_crispr_contig = crispr_operon_all_pos_in_crispr_contig.sort_values(by = 'leftmost_coord',
                                                                                                ascending = True)
        # merge further with positions of other None Cas proteins.
        protein_coding_records_df_contig = protein_coding_records_df.loc[
            protein_coding_records_df["crispr_contig_id"] == crispr_contig_id]
        crispr_operon_all_pos_in_crispr_contig = pd.concat(
            [crispr_operon_all_pos_in_crispr_contig, protein_coding_records_df_contig], axis = 0, sort = False).fillna(0)
        crispr_operon_all_pos_in_crispr_contig = crispr_operon_all_pos_in_crispr_contig.drop_duplicates(
            subset = ['leftmost_coord', 'rightmost_coord'],
            keep = "first")  # for the protein that are Cas, remove the duplicated records from protein coding results
        crispr_operon_all_pos_in_crispr_contig = crispr_operon_all_pos_in_crispr_contig.sort_values(by = 'leftmost_coord',
                                                                                                ascending = True)
        crispr_operon_all_pos_in_crispr_contig_list_frame.append(crispr_operon_all_pos_in_crispr_contig)
    return crispr_operon_all_pos_in_crispr_contig_list_frame


def read_protein_coding_records_of_crispr_array_containing_contigs(file_path):
    """
    Reads and processes a file containing protein-coding gene information for contigs that contain
    CRISPR arrays. It extracts details such as contig ID, protein ID, and positional coordinates,
    and returns a DataFrame.
    """
    result_list = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace from the entire line
            if line.startswith(">"):  # Check if the line starts with '>'
                # Split the line on spaces to extract the first word (ID part)
                parts = line[1:].split("#")
                parts = [part.strip() for part in parts]
                protein_id, leftmost_coord, rightmost_coord, strand = parts[0:4]
                crispr_contig_id = "_".join(protein_id.split("_")[:-1])
                result_list.append((crispr_contig_id, protein_id, leftmost_coord, rightmost_coord, strand))
    df = pd.DataFrame(result_list,
                      columns=['crispr_contig_id', 'protein_id', 'leftmost_coord', 'rightmost_coord', 'strand'])
    # Convert the coordinate and strand columns to integer types
    df['leftmost_coord'] = df['leftmost_coord'].astype(int)
    df['rightmost_coord'] = df['rightmost_coord'].astype(int)
    df['strand'] = df['strand'].astype(int)

    return df

def merge_crispr_operon_all_pos_in_crispr_contig_filter_by_cas_type(merged_crispr_array_record, flanking_size_record, cas_flanking_region, protein_fasta, pilercr_output_file, minced_output_file,
                                                                    cas_type = None):
    """
    Filters and merges information about CRISPR operons and their associated Cas proteins based
    on a specified Cas protein type, then organizes the data.
    """
    merged_crispr_array_record = pd.read_csv(merged_crispr_array_record, header = None, index_col = None,
                                             sep = ",")
    flanking_size_record = pd.read_csv(flanking_size_record, header = None, index_col = None,
                                       sep = "\t")

    protein_coding_records_df = read_protein_coding_records_of_crispr_array_containing_contigs(protein_fasta)
    protein_finding_join_cas_finding_flanking_region = pd.read_csv(
        cas_flanking_region, header = 0, index_col = None, sep = "\t", )

    # keep 'crispr_contig_ids' that contains  specific cas protein  type
    if cas_type is not None:
        crispr_ids_with_cas = protein_finding_join_cas_finding_flanking_region[
            protein_finding_join_cas_finding_flanking_region['hmm_id'].str.contains(cas_type, case = False)][
            'crispr_contig_id'].unique()
    else:
        crispr_ids_with_cas = protein_finding_join_cas_finding_flanking_region['crispr_contig_id'].unique()
    filtered_df = protein_finding_join_cas_finding_flanking_region[
        protein_finding_join_cas_finding_flanking_region['crispr_contig_id'].isin(crispr_ids_with_cas)]

    # For each cas variant, only keep one record of them (one cas protein can have multiple hmm profiles of sub-variants)
    unique_df = (
        filtered_df
        .groupby('crispr_contig_id', group_keys=False)[['crispr_contig_id', 'protein_id', 'leftmost_coord', 'rightmost_coord',
       'strand', 'hmm_id', 'full_seq_eval', 'full_seq_score']] # to avoid warning
        .apply(lambda x: x.sort_values(
            by='hmm_id',
            key=lambda col: col.str.contains(cas_type, case=False),  # Check if cas_type is in the string
            ascending=False  # Sort so that rows containing cas_type come first
        )
               # Sorting within each group:
               # by='hmm_id': Sorting is based on the 'hmm_id' column.
               # key=lambda col: col.str.contains(cas_type, case=False):
               # The str.contains() function checks if the string cas_type (a variable) is present in each entry of the 'hmm_id' column.
               # This returns a boolean Series (True or False), which is used as the sort key.
               # case=False: Ensures the check is case-insensitive.
               # ascending=False: Rows where 'hmm_id' contains cas_type appear first after sorting.
               .drop_duplicates(subset=['leftmost_coord', 'rightmost_coord'], keep='first'))
        .reset_index(drop=True)
    )

    # merge cas finding results  and crispr array results together and order them by position in the crispr_contig
    crispr_operon_all_pos_in_crispr_contig_list_frame = merge_crispr_operon_all_pos_in_crispr_contig(unique_df,
                                                                                                     merged_crispr_array_record,
                                                                                                     flanking_size_record,
                                                                                                     protein_coding_records_df,
                                                                                                     pilercr_output_file,
                                                                                                     minced_output_file)
    return crispr_operon_all_pos_in_crispr_contig_list_frame

def parallel_merge_crispr_operon_all_pos_in_crispr_contig_filter_by_cas_type(args):
    """
    Parallel processing of CRISPR operon and Cas protein positional data, filtering and
    merging the results for a specific Cas protein type if specified, and saving the
    results to a file.
    """
    merged_crispr_array_record, flanking_size_record, cas_flanking_region, protein_fasta, \
        pilercr_output_file, minced_output_file, cas_type = args
    try:
        crispr_operon_all_pos_in_crispr_contig_list_frame = merge_crispr_operon_all_pos_in_crispr_contig_filter_by_cas_type(
            merged_crispr_array_record, flanking_size_record, cas_flanking_region, protein_fasta, pilercr_output_file, minced_output_file,
            cas_type = cas_type
        )
    
        output_file = f"{flanking_size_record}2"
        if len(crispr_operon_all_pos_in_crispr_contig_list_frame) > 0:  # In case no results found
            crispr_operon_all_pos_in_crispr_contig_one_frame = pd.concat(crispr_operon_all_pos_in_crispr_contig_list_frame)
            crispr_operon_all_pos_in_crispr_contig_one_frame.to_csv(output_file, header = True, index = None, sep = "\t")
        else:
            pass

    except TypeError:
        print(f"block_pos_repeat_spacer from the sample with the file {protein_fasta} was empty. Continuing..")
        pass

##############################
# PAMpredict, phage database #
##############################
def check_blast_database(db_prefix, dbtype = "nucl"):
    """
    Check if blast database exist.
    """
    matching_files = glob.glob(f"{db_prefix}.*")
    if len(matching_files) > 1:
        return True
    else:
        return False

def run_makeblastdb(input_file):
    """
    Run makeblastdb (produces BLAST databases from FASTA files).
    """
    if not check_blast_database(input_file):
        try:
            # Define the command as a list of arguments
            command = ["makeblastdb", "-in", input_file, "-dbtype", "nucl"]
            print(" ".join(command))
            result = subprocess.run(command, check = True)

        except subprocess.CalledProcessError as e:
            print("An error occurred:", e.stderr)  # Display the error message from stderr
            raise  # Re-raise the exception if further handling is needed

def get_num_cpus():
    """
    Determines the number of CPUs that can be used for parallel processing in the current environment.
    """
    # Check if running under SLURM
    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        return int(slurm_cpus)

    # Fallback: Use os.cpu_count() if not under SLURM
    return int(os.cpu_count() / 2) or 1  # Default to 1 if os.cpu_count() fails

def reset_folder(folder):
    """
    Delete existing folder and recreate it.
    """
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def extract_base_filename(path):
    """Extract the base filename without .txt, _flanking_size.tsv, .tsv, .fasta or .faa suffix."""
    filename = os.path.basename(path)
    if filename.endswith('.txt'):
        return filename[:-4]
    elif filename.endswith('_flanking_size.tsv'):
        return filename[:-18]
    elif filename.endswith('.faa'):
        return filename[:-4]
    elif filename.endswith('.tsv'):
        return filename[:-4]
    elif filename.endswith('.fasta'):
        return filename[:-6]
    else:
        return filename


def group_and_filter_paths(list1, list2, list3, list4, list5, list6, list7):
    all_paths = list1 + list2 + list3 + list4 + list5 + list6 + list7
    grouped_paths = {}
    unique_files = []

    # Group paths by base name
    for path in all_paths:
        base_name = extract_base_filename(path)
        if base_name not in grouped_paths:
            grouped_paths[base_name] = []
        grouped_paths[base_name].append(path)

    # Keep only groups with all four files
    for base_name, paths in list(grouped_paths.items()):
        if len(paths) < 7:
            unique_files.extend(paths)
            del grouped_paths[base_name]

    # Create new lists with only shared paths
    new_list1 = sorted([path for path in list1 if extract_base_filename(path) in grouped_paths])
    new_list2 = sorted([path for path in list2 if extract_base_filename(path) in grouped_paths])
    new_list3 = sorted([path for path in list3 if extract_base_filename(path) in grouped_paths])
    new_list4 = sorted([path for path in list4 if extract_base_filename(path) in grouped_paths])
    new_list5 = sorted([path for path in list5 if extract_base_filename(path) in grouped_paths])
    new_list6 = sorted([path for path in list6 if extract_base_filename(path) in grouped_paths])
    new_list7 = sorted([path for path in list7 if extract_base_filename(path) in grouped_paths])

    return new_list1, new_list2, new_list3, new_list4, new_list5, new_list6, new_list7, unique_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Prepare PAM prediction data.")
    parser.add_argument('--phage_database', action = "store", dest = "phage_database")
    parser.add_argument('--cas_type', action = "store", dest = "cas_type")
    parser.add_argument('--cas_flanking_regions', action = "store", dest = "cas_flanking_regions", nargs='+')
    parser.add_argument('--num_processes', action = "store", dest = "num_processes", type = int)
    parser.add_argument('--output_directory', action = "store", dest = "output_directory")
    parser.add_argument('--pilercr_output_files', action = "store", dest = "pilercr_output_files", nargs='+')
    parser.add_argument('--minced_output_files', action = "store", dest = "minced_output_files", nargs='+')
    parser.add_argument('--protein_fastas', action = "store", dest = "protein_fastas", nargs='+')
    parser.add_argument('--merged_crispr_array_records', action = "store", dest = "merged_crispr_array_records", nargs='+')
    parser.add_argument('--flanking_size_records', action = "store", dest = "flanking_size_records", nargs='+')
    parser.add_argument('--merged_fastas', action="store", dest="merged_fastas", nargs='+')

    args = parser.parse_args()

    phage_database = args.phage_database
    cas_type = args.cas_type
    tsv_input = args.cas_flanking_regions
    num_processes = args.num_processes
    output_directory = args.output_directory
    pilercr_output_files = args.pilercr_output_files
    minced_output_files = args.minced_output_files
    protein_fastas = args.protein_fastas
    flanking_size_records = args.flanking_size_records
    merged_crispr_array_records = args.merged_crispr_array_records
    merged_fastas = args.merged_fastas

    # only files that exist
    flanking_size_records = " ".join([f for f in list(set(flanking_size_records)) if os.path.exists(f) and os.path.getsize(f) > 0])
    merged_crispr_array_records = " ".join([f for f in list(set(merged_crispr_array_records)) if os.path.exists(f) and os.path.getsize(f) > 0])
    protein_fastas = " ".join([f for f in list(set(protein_fastas)) if os.path.exists(f) and os.path.getsize(f) > 0])
    cas_flanking_regions = " ".join([f for f in list(set(tsv_input)) if os.path.exists(f) and os.path.getsize(f) > 0])
    pilercr_output_files = " ".join([f for f in list(set(pilercr_output_files)) if os.path.exists(f) and os.path.getsize(f) > 0])
    minced_output_files = " ".join([f for f in list(set(minced_output_files)) if os.path.exists(f) and os.path.getsize(f) > 0])
    merged_fastas = " ".join([f for f in list(set(merged_fastas)) if os.path.exists(f) and os.path.getsize(f) > 0])

    flanking_size_records = flanking_size_records.split(" ")
    merged_crispr_array_records = merged_crispr_array_records.split(" ")
    protein_fastas = protein_fastas.split(" ")
    cas_flanking_regions = cas_flanking_regions.split(" ")
    pilercr_output_files = pilercr_output_files.split(" ")
    minced_output_files = minced_output_files.split(" ")
    merged_fastas = merged_fastas.split(" ")

    ATCG_mapping = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N", }

    phage_database_list = [(pathlib.Path(phage_database).parent.absolute(), phage_database), ]

    protein_coding_genes_folder = f"{output_directory}/merged/"
    crispr_operon_filtered_results_folder = f"{output_directory}/merged/"

    crispr_operon_all_pos_in_crispr_contig_folder = f"{output_directory}/{cas_type}/crispr_operon_filtered_all_pos_in_crispr_contig_one_frame_{cas_type}/"
    all_cas_proteins_folder = f"{output_directory}/{cas_type}/all_{cas_type}_proteins/"
    cas_clustering_folder = f"{all_cas_proteins_folder}mmseqs2_clustering/"
    cas_clustering_split_folder = f"{cas_clustering_folder}/{cas_type}/{cas_type}_cluster_split/"

    script_name = os.path.basename(__file__)

    logging.info(f"start")

    # everytime for now cas cluster results , we need to create new folder for new results
    pam_prediction_output_directory = f"{output_directory}PAM_prediction_output/"
    repeat_aggregate_folder = f"{cas_clustering_folder}repeat_aggregate/"
    spacer_aggregate_folder = f"{cas_clustering_folder}spacer_aggregate/"
    cdhit_repeat_clustering_folder = f"{cas_clustering_folder}cdhit_repeat_clustering/"
    repeat_oriented_spacer_aggregate_folder = f"{cas_clustering_folder}repeat_oriented_spacer_aggregate/"
    cdhit_repeat_oriented_spacer_clustering_folder = f"{cas_clustering_folder}cdhit_repeat_oriented_spacer_clustering/"

    create_folder_if_not_existing(all_cas_proteins_folder)
    create_folder_if_not_existing(cas_clustering_folder)
    create_folder_if_not_existing(cas_clustering_split_folder)

    # Check if running under SLURM and get the number of CPUs
    num_cpus = get_num_cpus()
    if "SLURM_JOB_ID" in os.environ:
        logging.info(f"Running on SLURM with job ID: {os.environ['SLURM_JOB_ID']}")
    else:
        num_cpus = int(num_cpus / 10)
    logging.info(f"num_cpus:{num_cpus}")

    # collect all crispr operon(repeat and spacers) , cas protein and other proteins (for quality check and filtering) into one file for later use
    crispr_operon_filtered_results_files = get_existing_file(cas_flanking_regions, return_only_non_empty_file = True)
    crispr_operon_filtered_results_files = sorted(list(crispr_operon_filtered_results_files))  # to keep same order from different runs

    # Run the grouping and filtering
    merged_crispr_array_records, flanking_size_records, cas_flanking_regions, protein_fastas, pilercr_output_files, minced_output_files, merged_fastas, unique_files = group_and_filter_paths(merged_crispr_array_records, flanking_size_records, cas_flanking_regions, protein_fastas, pilercr_output_files, minced_output_files, merged_fastas)

    print("\nUnique files (removed with warnings):")
    for path in unique_files:
        print(f"  Warning: {path} does not have all other needed files and has been removed.")

    mp_args = [
        (mcr, fsr, cfr, pf, pof, mof, cas_type)
        for mcr, fsr, cfr, pf, pof, mof in zip(sorted(merged_crispr_array_records), sorted(flanking_size_records), sorted(cas_flanking_regions), sorted(protein_fastas), sorted(pilercr_output_files), sorted(minced_output_files)) # sort alphabetically
    ]

    with Pool(processes = num_cpus) as pool:
        pool.map(parallel_merge_crispr_operon_all_pos_in_crispr_contig_filter_by_cas_type, mp_args)
    logging.info(f"parallel_merge_crispr_operon_all_pos_in_crispr_contig_filter_by_cas_type done")

    # write all cas protein to one file , later to be used  for clustering
    # this is bottleneck, as we need to copy all file into one file sequentially, won't make things faster
    collect_all_cas_in_one_file(sorted(flanking_size_records), all_cas_proteins_folder, cas_type, sorted(merged_fastas))
    logging.info(f"collect_all_cas_in_one_file done")

    # cluster all cas proteins
    result = mmseqs_easy_clustering(all_cas_proteins_folder, cas_type, num_cpus)
    print(result)
    if result.returncode == 0:
        logging.info(f"mmseqs_easy_clustering done")
    else:
        logging.info(f"mmseqs_easy_clustering failed")
        logging.info(f"Command failed with return code {result.returncode}")
        logging.info(f"Error message: {result.stderr}")

    # now split results to different files to be process later in parallel
    cas_clustering_results = pd.read_csv(
        f"{cas_clustering_folder}{cas_type}_cluster_cluster.tsv",
        header=None, index_col=None, sep="\t")
    grouped = cas_clustering_results.groupby(0, sort=False)
    grouped_list = list(grouped)
    chunk_size = len(grouped_list) // num_processes
    logging.info(
        f"cas_clustering_results.shape:{cas_clustering_results.shape};len(grouped_list):{len(grouped_list)}; chunk_size:{chunk_size}; num_process:{num_processes}")
    for i in range(num_processes):
        start_index = i * chunk_size
        end_index = (i + 1) * chunk_size if i != num_processes - 1 else len(grouped_list)
        chunk = grouped_list[start_index:end_index]
        chunk_df = pd.concat([group for _, group in chunk])
        chunk_df.to_csv(f"{cas_clustering_split_folder}{cas_type}_clustering_results_part{i}.tsv",
                        sep="\t", header=False, index=False)

    logging.info(f"split cas_clustering_results done")

    logging.info(f"Clean up and create new folders done")

    # prepare phage database
    for _, phage_fasta in phage_database_list:
        run_makeblastdb(phage_fasta)

    logging.info(f"Finished")