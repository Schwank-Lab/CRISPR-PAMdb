import subprocess
import re
import os
from Bio import SeqIO
import pandas as pd
from collections import Counter
import logging
import argparse

ATCG_mapping = {"A": "T", "T": "A", "C": "G", "G": "C"}

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

# using cas protein clustering results to aggregate related repeat or spacers  for near identical cas protein  and
########################################################
def get_repeat_or_spacer_seq_from_cas_member_id(cas_member_id, label, tsv_file, merged_fasta):
    """
    Get repeat or spacer sequence from a given cas member ID.
    """
    set_id = cas_member_id.split("#")[0]
    protein_id = cas_member_id.split("#")[1]
    crispr_contig_id = protein_id[:protein_id.rfind("_")]

    crispr_operon_all_pos_in_crispr_contig_result = pd.read_csv(tsv_file, header=0, index_col=None, sep="\t")
    crispr_core_operon_pos = crispr_operon_all_pos_in_crispr_contig_result[~(
                (crispr_operon_all_pos_in_crispr_contig_result["hmm_id"] == "0") & (
                    crispr_operon_all_pos_in_crispr_contig_result["protein_id"] != "repeat") & (
                            crispr_operon_all_pos_in_crispr_contig_result["protein_id"] != "spacer"))]
    crispr_core_operon_pos_cur_contig = crispr_core_operon_pos.loc[
        crispr_core_operon_pos["crispr_contig_id"] == crispr_contig_id]

    # get repeat seq or spacer
    repeat_list = crispr_core_operon_pos_cur_contig.loc[
        crispr_core_operon_pos_cur_contig["protein_id"] == label].values.tolist()
    repeat_seq = list()
    for record in repeat_list:
        crispr_contig_id, leftmost_coord, rightmost_coord, strand = record[0], int(record[2]), int(record[3]), record[4]
        repeat_seq.append((cas_member_id,
                           return_seq_from_crispr_contig(merged_fasta, crispr_contig_id, leftmost_coord, rightmost_coord,
                                                         strand, tsv_file)))

    return (repeat_seq)


def consensus_repeat(repeat_seq):
    """
    Create a consensus sequence from a list of repeat sequences.
    """
    contig_id = repeat_seq[0][0]
    strings = [str(record[1]) for record in repeat_seq]
    # Transpose the list of strings to get columns instead of rows
    columns = zip(*strings)

    # For each column, find the most common character and join them into the consensus string
    consensus = ''.join(Counter(column).most_common(1)[0][0] for column in columns)

    return [(contig_id, consensus)]


def write_repeat_or_spacer_seq(cas_cluster_id, seq_list, output_folder, label, write_mode="w"):
    """
    Write repeat or spacer sequences to a FASTA file.
    """
    file_name = f"{output_folder}{os.path.basename(seq_list[0][0][0]).replace('.tsv', '')}.fasta"
    os.makedirs(output_folder, exist_ok=True)
    with open(file_name, write_mode) as fasta_file:
        for sub_seq_list in seq_list:
            count = 0
            for cas_member_id, seq in sub_seq_list:
                fasta_file.write(f">{os.path.basename(cas_member_id).replace('.tsv', '')}{label}{count}\n")  # Write the sequence ID as a header
                fasta_file.write(f"{seq}\n")  # Write the sequence
                count += 1


def collect_repeat_spacer_seq_by_cas_clustering_per_cluster(cas_cluster_id, group, tsv_file,
                                                            merged_fasta, repeat_aggregate_folder,
                                                            spacer_aggregate_folder):
    """
    Collect repeat and spacer sequences for a given cas cluster.
    """
    group_list = group.values.tolist()
    repeat_consensus_seq_list = []
    spacer_seq_list = []

    for _, cas_member_id in group_list:
        # Get repeat sequences
        repeat_seq = get_repeat_or_spacer_seq_from_cas_member_id(cas_member_id, "repeat", tsv_file, merged_fasta)
        if len(repeat_seq) > 0:
            repeat_consensus_seq = consensus_repeat(repeat_seq)  # Create consensus repeat
            repeat_consensus_seq_list.append(repeat_consensus_seq)

        # Get spacer sequences
        spacer_seq_list.append(
            get_repeat_or_spacer_seq_from_cas_member_id(cas_member_id, "spacer", tsv_file, merged_fasta)
        )

    # Write repeat and spacer sequences
    print(repeat_consensus_seq_list)
    write_repeat_or_spacer_seq(cas_cluster_id, repeat_consensus_seq_list, repeat_aggregate_folder, "repeat")
    write_repeat_or_spacer_seq(cas_cluster_id, spacer_seq_list, spacer_aggregate_folder, "spacer")


# CD-HIT to cluster repeat to consistently orient crispr array/spacer direction
########################################################
def cd_hit_est_repeat_clustering(input_folder, output_folder, cas_cluster_id):
    """
    CD-HIT to cluster repeat sequences.
    """
    ifile_name = f"{input_folder}{cas_cluster_id}.fasta"
    ofile_name = f"{output_folder}{cas_cluster_id}.fasta"

    os.makedirs(output_folder, exist_ok=True)
    skw = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sub_cmd = ["cd-hit-est", "-i", ifile_name, "-o", ofile_name, "-c", "0.8", "-s", "0.75", "-d",
               "0"]  # "-c", "0.9", "-s","0.9"   "-c", "0.9", "-s","0.8"  "-c", "0.8", "-s","0.8",  "-c", "0.8", "-s","0.75","-d","0"
    p = subprocess.run(sub_cmd,
                       universal_newlines=True, **skw)

def parse_cluster_data(cas_cluster_id, cdhit_repeat_clustering_folder):
    """
    Parse the CD-HIT clustering results to extract repeat clusters.
    """
    filename = f"{cdhit_repeat_clustering_folder}{cas_cluster_id}.fasta.clstr"
    clusters = dict()
    current_key = None  # To store the current cluster key

    if os.path.exists(filename):
        with open(filename, "r") as ifile:
            for line in ifile:
                line = line.strip()
                # Check if the line defines a new cluster
                if line.startswith(">Cluster"):
                    current_key = line[-1]
                    clusters[current_key] = list()

                elif line.endswith("*"):
                    match = re.search(r'>(\S+?)\.\.\.', line)
                    id_ = match.group(1)
                    clusters[current_key].append((id_, "+", "*"))
                else:
                    # Extract ID and orientation from non-cluster lines
                    match = re.search(r'>(\S+?)\.\.\. at ([+-])/(\d+\.\d+)%', line)
                    id_ = match.group(1)
                    orientation = match.group(2)
                    percentage = match.group(3)
                    clusters[current_key].append((id_, orientation, percentage))

        return clusters
    return None


def longest_list_info(d):
    """
    Find the longest list in a dictionary and return its key, length, and total length of all lists.
    """
    longest_key = None
    longest_length = 0
    total_length = 0

    # Iterate through the dictionary
    for key, value in d.items():
        # Get the length of the current list
        current_length = len(value)
        total_length += current_length  # Sum of all list lengths

        # Check if this is the longest list found so far
        if current_length > longest_length:
            longest_length = current_length
            longest_key = key

    return longest_key, longest_length, total_length


def get_repeat_representative_dict_from_cas_clustering(repeat_clustering_results):
    """
    Get the representative repeat cluster from the clustering results.
    """
    if repeat_clustering_results:
        longest_key, longest_length, total_length = longest_list_info(repeat_clustering_results)

        repeat_clustering_results_frame = pd.DataFrame(
            [(repeat_id.split("repeat")[0], repeat_id, direction, percentage) for repeat_id, direction, percentage in
             repeat_clustering_results[longest_key]])
        consistent_values = repeat_clustering_results_frame.groupby(0)[
                                2].nunique() == 1  # check if for a same cirspr operon. the repeat are in same direction , this is not needed after use consensure repeat for clustering
        consistent_repeat_direction = consistent_values.all()
        assert consistent_repeat_direction

        repeat_representative_frame = repeat_clustering_results_frame[[0, 2]]
        repeat_representative_frame = repeat_representative_frame.drop_duplicates()

        repeat_representative_dict = repeat_representative_frame.values.tolist()
        repeat_representative_dict = {id: dire for id, dire in repeat_representative_dict}

        return (longest_key, longest_length, total_length, repeat_representative_dict)
    return None, None, None, None


def orient_spacer(seq, orientation):
    """
    Orient the spacer sequence based on the given orientation.
    """
    assert (orientation == "-") or (orientation == "+")
    if orientation == "-":
        seq = "".join([ATCG_mapping.get(s, "N") for s in
                       seq])  # this will get (complimentary) seq on the opposite strand from left to right
        seq = seq[::-1]  # this will get seq on the opposite strand from right to left, same as results from prodigal
    else:
        seq = "".join([s if s in ATCG_mapping else "N" for s in
                       seq])  # change non-ATCG characters to N, as later PAMpredict only allow Base ATCGN
    return seq


def orient_spacer_by_repeat_cdhit_results(cas_cluster_id, repeat_representative_dict, spacer_aggregate_folder,
                                          repeat_oriented_spacer_aggregate_folder):
    """
    Orient spacers based on the repeat direction from CD-HIT clustering results.
    """
    ifile_name = f"{spacer_aggregate_folder}{cas_cluster_id}.fasta"
    ofile_name = f"{repeat_oriented_spacer_aggregate_folder}{cas_cluster_id}.fasta"
    os.makedirs(repeat_oriented_spacer_aggregate_folder, exist_ok=True)

    orient_record = []
    skip_next_line = False  # Flag to indicate if the next line should be skipped

    with open(ifile_name, "r") as ifile:
        for line in ifile:
            line = line.strip()

            # If skip_next_line is set, skip this line and reset the flag
            if skip_next_line:
                skip_next_line = False
                continue

            if line.startswith(">"):
                cas_protein_plus_spacer_id = line[1:]
                cas_protein_id = cas_protein_plus_spacer_id.split("spacer")[0]
                # Check if cas_protein_id is in repeat_representative_dict
                if cas_protein_id not in repeat_representative_dict:
                    skip_next_line = True  # Set flag to skip the next line
                    continue  # Skip the current line with ">"
                repeat_dire = repeat_representative_dict[cas_protein_id]
            else:
                seq = orient_spacer(line, repeat_dire)
                orient_record.append((cas_protein_plus_spacer_id, seq))

    with open(ofile_name, "w") as fasta_file:
        for seq_id, seq in orient_record:
            fasta_file.write(f">{seq_id}\n")
            fasta_file.write(f"{seq}\n")


def cd_hit_est_spacer_clustering(input_folder, output_folder, cas_cluster_id):
    """
    CD-HIT to cluster oriented spacers to remove duplicates and get representative sequences.
    """
    ifile_name = f"{input_folder}{cas_cluster_id}.fasta"
    ofile_name = f"{output_folder}{cas_cluster_id}.fasta"
    os.makedirs(output_folder, exist_ok=True)

    skw = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sub_cmd = ["cd-hit-est", "-i", ifile_name, "-o", ofile_name, "-c", "0.95", "-s", "1.0", "-r", "0", "-d",
               "0"]  # set r to 0 as we already use repeat clustering results to set then in consistent orientation
    # -r   1 or 0, default 1, by default do both +/+ & +/- alignments
    #         if set to 0, only +/+ strand alignment
    p = subprocess.run(sub_cmd,
                       universal_newlines=True, **skw)  # , **skw, add this to silicene output


def process_each_cas_cluster(mp_args):  # define function here so all variables in the main function is accessible
    """
    Process a single cas cluster. Encapsulates all steps for processing one group.
    """

    (cas_cluster_id, group, cas_type, all_cas_proteins_folder, cas_clustering_folder,
     merged_directory, repeat_aggregate_folder, spacer_aggregate_folder,
     cdhit_repeat_clustering_folder,
     repeat_oriented_spacer_aggregate_folder,
     cdhit_repeat_oriented_spacer_clustering_folder) = mp_args
    tsv_file = cas_cluster_id.split("#")[0].replace('.tsv', '_flanking_size.tsv2')
    merged_file = cas_cluster_id.split("#")[0].replace('.tsv', '.fasta')
    cas_cluster_id = os.path.basename(cas_cluster_id).replace('.tsv', '')

    try:
        # using cas protein clustering results to aggregate related repeat or spacers  for near identical cas protein)
        collect_repeat_spacer_seq_by_cas_clustering_per_cluster(cas_cluster_id, group,
                                                                tsv_file,
                                                                merged_file, repeat_aggregate_folder,
                                                                spacer_aggregate_folder)

        # CD-HIT to cluster repeat to consistently orient crispr array/spacer direction
        cd_hit_est_repeat_clustering(repeat_aggregate_folder, cdhit_repeat_clustering_folder, cas_cluster_id)
        repeat_clustering_results = parse_cluster_data(cas_cluster_id, cdhit_repeat_clustering_folder)

        if repeat_clustering_results:
            # Get representative repeat cluster
            longest_key, longest_length, total_length, repeat_representative_dict = get_repeat_representative_dict_from_cas_clustering(
                repeat_clustering_results)
            repeat_clustering_representative_ratio = (
            cas_cluster_id, longest_length, total_length, longest_length / total_length)

            # Orient spacers
            orient_spacer_by_repeat_cdhit_results(cas_cluster_id, repeat_representative_dict, spacer_aggregate_folder,
                                                  repeat_oriented_spacer_aggregate_folder)

            # Remove duplicate spacers
            cd_hit_est_spacer_clustering(repeat_oriented_spacer_aggregate_folder,
                                         cdhit_repeat_oriented_spacer_clustering_folder, cas_cluster_id)
            return repeat_clustering_representative_ratio
    except Exception as e:
        print(f"Error processing {cas_cluster_id}: {e}")
        raise



def main():
    parser = argparse.ArgumentParser(description="Script to process cas clusters and aggregate repeat/spacer sequences for pam prediction.")
    parser.add_argument("--output_directory", type=str, required=True, help="Output folder path")
    parser.add_argument("--cas_type", type=str, default="cas9", help="Cas type (default: cas9)")
    parser.add_argument("--sublist", type=str, help="Path to sublist of cas clusters to process")
    parser.add_argument("--repeat_clustering_representative_ratio", type=str, required=True, help="repeat_clustering_representative_ratio")
    parser.add_argument("--prepare_spacers_directory", type=str, required=True)

    args = parser.parse_args()
    output_directory = args.output_directory
    cas_type = args.cas_type
    sublist = args.sublist
    repeat_clustering_representative_ratio = args.repeat_clustering_representative_ratio

    all_cas_proteins_folder = f"{output_directory}/{cas_type}/all_{cas_type}_proteins/"
    cas_clustering_folder = f"{all_cas_proteins_folder}mmseqs2_clustering/"

    repeat_aggregate_folder = f"{cas_clustering_folder}repeat_aggregate/"
    spacer_aggregate_folder = f"{cas_clustering_folder}spacer_aggregate/"
    cdhit_repeat_clustering_folder = f"{cas_clustering_folder}cdhit_repeat_clustering/"
    repeat_oriented_spacer_aggregate_folder = f"{cas_clustering_folder}repeat_oriented_spacer_aggregate/"
    cdhit_repeat_oriented_spacer_clustering_folder = f"{cas_clustering_folder}cdhit_repeat_oriented_spacer_clustering/"
    merged_directory = f"{output_directory}/merged/"
    os.makedirs(cdhit_repeat_clustering_folder, exist_ok=True)

    logging.info(f"Starting preparing spacers")
    cas_clustering_results_i = pd.read_csv(sublist, sep="\t", header=None,index_col=None)
    logging.info(f"cas_clustering_results_i.shape:{cas_clustering_results_i.shape}")

    grouped = cas_clustering_results_i.groupby(0, sort=False)
    mp_args = [(cas_cluster_id, group, cas_type, all_cas_proteins_folder, cas_clustering_folder,
                merged_directory, repeat_aggregate_folder, spacer_aggregate_folder,
                cdhit_repeat_clustering_folder, repeat_oriented_spacer_aggregate_folder,
                cdhit_repeat_oriented_spacer_clustering_folder) for cas_cluster_id, group in grouped]
    logging.info(f"len(mp_args):{len(mp_args)}")

    results = []
    for args in mp_args:
        result = process_each_cas_cluster(args)
        results.append(result)

    repeat_clustering_representative_ratio_list = [res for res in results if res is not None]
    repeat_clustering_representative_ratio_frame = pd.DataFrame(repeat_clustering_representative_ratio_list,
                                                                columns=["cas_cluster_id", "main_cluster_size",
                                                                         "all_cluster_size", "ratio"])
    repeat_clustering_representative_ratio_frame.to_csv(repeat_clustering_representative_ratio,
                header=True, index=None, sep="\t")

    logging.info(f"finished")


if __name__ == "__main__":
    main()