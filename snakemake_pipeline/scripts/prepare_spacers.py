import os
import sys
import subprocess
import re
import SeqIO
import pandas as pd
from collections import Counter

import logging

# Add the project root to the system path
project_root = os.path.dirname(os.path.abspath(__file__)) + '/../'
sys.path.append(project_root)
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
import argparse

ATCG_mapping = {"A": "T", "T": "A", "C": "G", "G": "C"}

def return_seq_from_crisprcontig(set_id,crispr_contig_id,start_pos,end_pos,strand,Merged_output_folder):
    try:
        crispr_conitg_seq_current=SeqIO.to_dict(SeqIO.parse(f"{Merged_output_folder}{set_id}.fasta", "fasta"))[crispr_contig_id]
        return_seq=crispr_conitg_seq_current[start_pos-1:end_pos].seq
        if strand == -1:
            return_seq="".join([ATCG_mapping[s] for s in return_seq])  # this will get seq on the opposite strand from left to right
            return_seq=return_seq[::-1] # this will get seq on the opposite strand from right to left, same as results from prodigal
        return(return_seq)
    except Exception as e:
        # Capture and print all exceptions
        print(f"An error occurred: {e};{set_id};{crispr_contig_id}")
        raise

# using cas protein clustering results to aggregate related repeat or spacers  for near identical cas protein  and
########################################################
def get_repeatorspacer_seq_from_cas_member_id(cas_member_id, label, Crispr_operon_allpos_in_crisprcontig_folder,
                                              Merged_output_folder):
    set_id = cas_member_id.split("#")[0]
    protein_id = cas_member_id.split("#")[1]
    crispr_contig_id = protein_id[:protein_id.rfind("_")]

    Crispr_operon_allpos_in_crisprcontig_result = pd.read_csv(
        f"{Crispr_operon_allpos_in_crisprcontig_folder}{set_id}.tsv", header=0, index_col=None, sep="\t")
    Crispr_coreOperon_pos = Crispr_operon_allpos_in_crisprcontig_result[~(
                (Crispr_operon_allpos_in_crisprcontig_result["hmm_id"] == "0") & (
                    Crispr_operon_allpos_in_crisprcontig_result["protein_id"] != "repeat") & (
                            Crispr_operon_allpos_in_crisprcontig_result["protein_id"] != "spacer"))]
    Crispr_coreOperon_pos_curcontig = Crispr_coreOperon_pos.loc[
        Crispr_coreOperon_pos["crispr_contig_id"] == crispr_contig_id]

    # get repeat seq or spacer
    repeat_list = Crispr_coreOperon_pos_curcontig.loc[
        Crispr_coreOperon_pos_curcontig["protein_id"] == label].values.tolist()
    repeat_seq = list()
    for record in repeat_list:
        crispr_contig_id, leftmost_coord, rightmost_coord, strand = record[0], int(record[2]), int(record[3]), record[4]
        repeat_seq.append((cas_member_id,
                           return_seq_from_crisprcontig(set_id, crispr_contig_id, leftmost_coord, rightmost_coord,
                                                        strand, Merged_output_folder)))

    return (repeat_seq)


def consensus_repeat(repeat_seq):
    contig_id = repeat_seq[0][0]
    strings = [str(record[1]) for record in repeat_seq]
    # Transpose the list of strings to get columns instead of rows
    columns = zip(*strings)

    # For each column, find the most common character and join them into the consensus string
    consensus = ''.join(Counter(column).most_common(1)[0][0] for column in columns)

    return [(contig_id, consensus)]


def write_repeat_or_spacer_seq(cas_cluster_id, seq_list, output_folder, label, write_mode="w"):
    file_name = f"{output_folder}{cas_cluster_id}.fasta"
    with open(file_name, write_mode) as fasta_file:
        for sub_seq_list in seq_list:
            count = 0
            for cas_member_id, seq in sub_seq_list:
                fasta_file.write(f">{cas_member_id}{label}{count}\n")  # Write the sequence ID as a header
                fasta_file.write(f"{seq}\n")  # Write the sequence
                count += 1


def collect_repeat_spacer_seq_by_cas_clustering_percluser(cas_cluster_id, group,
                                                          Crispr_operon_allpos_in_crisprcontig_folder,
                                                          Merged_output_folder, repeat_aggregate_folder,
                                                          spacer_aggregate_folder):
    group_list = group.values.tolist()
    repeat_consensus_seq_list = []
    spacer_seq_list = []

    for _, cas_member_id in group_list:
        # Get repeat sequences
        repeat_seq = get_repeatorspacer_seq_from_cas_member_id(
            cas_member_id, "repeat", Crispr_operon_allpos_in_crisprcontig_folder, Merged_output_folder
        )
        repeat_consensus_seq = consensus_repeat(repeat_seq)  # Create consensus repeat
        repeat_consensus_seq_list.append(repeat_consensus_seq)

        # Get spacer sequences
        spacer_seq_list.append(
            get_repeatorspacer_seq_from_cas_member_id(
                cas_member_id, "spacer", Crispr_operon_allpos_in_crisprcontig_folder, Merged_output_folder
            )
        )

    # Write repeat and spacer sequences
    print(repeat_consensus_seq_list)
    write_repeat_or_spacer_seq(cas_cluster_id, repeat_consensus_seq_list, repeat_aggregate_folder, "repeat")
    write_repeat_or_spacer_seq(cas_cluster_id, spacer_seq_list, spacer_aggregate_folder, "spacer")


# CD-HIT to cluster repeat to consistently orient crispr array/spacer direction
########################################################
def cd_hit_est_repeat_clustering(input_folder, output_folder, cas_cluster_id):
    ifile_name = f"{input_folder}{cas_cluster_id}.fasta"
    ofile_name = f"{output_folder}{cas_cluster_id}.fasta"

    skw = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sub_cmd = ["cd-hit-est", "-i", ifile_name, "-o", ofile_name, "-c", "0.8", "-s", "0.75", "-d",
               "0"]  # "-c", "0.9", "-s","0.9"   "-c", "0.9", "-s","0.8"  "-c", "0.8", "-s","0.8",  "-c", "0.8", "-s","0.75","-d","0"
    p = subprocess.run(sub_cmd,
                       universal_newlines=True, **skw)

def parse_cluster_data(cas_cluster_id, cdhit_repeat_clustering_folder):
    filename = f"{cdhit_repeat_clustering_folder}{cas_cluster_id}.fasta.clstr"
    clusters = dict()
    current_key = None  # To store the current cluster key

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


def longest_list_info(d):
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


def orient_spacer(seq, orientation):
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
    ifile_name = f"{spacer_aggregate_folder}{cas_cluster_id}.fasta"
    ofile_name = f"{repeat_oriented_spacer_aggregate_folder}{cas_cluster_id}.fasta"

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
                cas_protein_plusspacer_id = line[1:]
                cas_protein_id = cas_protein_plusspacer_id.split("spacer")[0]
                # Check if cas_protein_id is in repeat_representative_dict
                if cas_protein_id not in repeat_representative_dict:
                    skip_next_line = True  # Set flag to skip the next line
                    continue  # Skip the current line with ">"
                repeat_dire = repeat_representative_dict[cas_protein_id]
            else:
                seq = orient_spacer(line, repeat_dire)
                orient_record.append((cas_protein_plusspacer_id, seq))

    with open(ofile_name, "w") as fasta_file:
        for seq_id, seq in orient_record:
            fasta_file.write(f">{seq_id}\n")
            fasta_file.write(f"{seq}\n")


def cd_hit_est_spacer_clustering(input_folder, output_folder, cas_cluster_id):
    ifile_name = f"{input_folder}{cas_cluster_id}.fasta"
    ofile_name = f"{output_folder}{cas_cluster_id}.fasta"

    skw = dict(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sub_cmd = ["cd-hit-est", "-i", ifile_name, "-o", ofile_name, "-c", "0.95", "-s", "1.0", "-r", "0", "-d",
               "0"]  # set r to 0 as we already use repeat clustering results to set then in consistent orientation
    # -r   1 or 0, default 1, by default do both +/+ & +/- alignments
    #         if set to 0, only +/+ strand alignment
    p = subprocess.run(sub_cmd,
                       universal_newlines=True, **skw)  # , **skw, add this to silicene output


def process_each_cas_cluster(mp_args):  # define function here so all variables in the main function is accessable
    """
    Process a single cas cluster. Encapsulates all steps for processing one group.
    """

    (cas_cluster_id, group, cas_type, All_cas_proteins_folder, cas_clustering_folder,
     Crispr_operon_allpos_in_crisprcontig_folder,
     Merged_output_folder, repeat_aggregate_folder, spacer_aggregate_folder,
     cdhit_repeat_clustering_folder,
     repeat_oriented_spacer_aggregate_folder,
     cdhit_repeatoriented_spacer_clustering_folder) = mp_args

    repeat_clustering_representative_ratio = None

    try:

        # using cas protein clustering results to aggregate related repeat or spacers  for near identical cas proteinr)
        collect_repeat_spacer_seq_by_cas_clustering_percluser(cas_cluster_id, group,
                                                              Crispr_operon_allpos_in_crisprcontig_folder,
                                                              Merged_output_folder, repeat_aggregate_folder,
                                                              spacer_aggregate_folder)

        # CD-HIT to cluster repeat to consistently orient crispr array/spacer direction
        cd_hit_est_repeat_clustering(repeat_aggregate_folder, cdhit_repeat_clustering_folder, cas_cluster_id)
        repeat_clustering_results = parse_cluster_data(cas_cluster_id, cdhit_repeat_clustering_folder)

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
                                     cdhit_repeatoriented_spacer_clustering_folder, cas_cluster_id)



    except Exception as e:
        print(f"Error processing {cas_cluster_id}: {e}")
        raise

    return repeat_clustering_representative_ratio


def main():
    parser = argparse.ArgumentParser(description="PAM Identification prepareSpacers Script")
    parser.add_argument("--output_folder", type=str, required=True, help="Output folder path")
    parser.add_argument("--cas_type", type=str, default="cas9", help="CAS type (default: cas9)")
    parser.add_argument("--sublist_id", type=int, default=0, help="Sublist ID (default: 0)")

    args = parser.parse_args()

    output_folder = args.output_folder
    cas_type = args.cas_type
    sublist_id = args.sublist_id

    Merged_output_folder = f"{output_folder}merge/"
    Crispr_operon_allpos_in_crisprcontig_folder = f"{output_folder}/{cas_type}/crispr_operon_filtered_all_pos_in_crispr_contig_one_frame_{cas_type}/"

    All_cas_proteins_folder = f"{output_folder}/{cas_type}/all_{cas_type}_proteins/"  # these folder created alreay in "PAM_identification_prepareData.py"
    cas_clustering_folder = f"{All_cas_proteins_folder}mmseqs2_clustering/"
    cas_clustering_splited_folder = f"{cas_clustering_folder}/{cas_type}/{cas_type}_cluster_split/"

    repeat_aggregate_folder = f"{cas_clustering_folder}repeat_aggregate/"
    spacer_aggregate_folder = f"{cas_clustering_folder}spacer_aggregate/"
    cdhit_repeat_clustering_folder = f"{cas_clustering_folder}cdhit_repeat_clustering/"
    repeat_oriented_spacer_aggregate_folder = f"{cas_clustering_folder}repeat_oriented_spacer_aggregate/"
    cdhit_repeatoriented_spacer_clustering_folder = f"{cas_clustering_folder}cdhit_repeat_oriented_spacer_clustering/"

    script_name = os.path.basename(__file__)
    logging.basicConfig(filename=f'{cas_clustering_splited_folder}{script_name}_{sublist_id}.log',
                        format="%(asctime)s;%(levelname)s;%(message)s", level=logging.INFO)
    logging.info(f"start")

    # cas_clustering_results=pd.read_csv(f"{cas_clustering_folder}{cas_type}_cluster_cluster.tsv",header=None,index_col=None,sep="\t") # get from above, collect_repeat_spacer_seq_by_cas_clustering()
    cas_clustering_results_i = pd.read_csv(
        f"{cas_clustering_splited_folder}{cas_type}_clustering_results_part{sublist_id}.tsv", sep="\t", header=None,
        index_col=None)
    logging.info(f"cas_clustering_results_i.shape:{cas_clustering_results_i.shape}")

    grouped = cas_clustering_results_i.groupby(0, sort=False)
    mp_args = [(cas_cluster_id, group, cas_type, All_cas_proteins_folder, cas_clustering_folder,
                Crispr_operon_allpos_in_crisprcontig_folder,
                Merged_output_folder, repeat_aggregate_folder, spacer_aggregate_folder,
                cdhit_repeat_clustering_folder,
                repeat_oriented_spacer_aggregate_folder,
                cdhit_repeatoriented_spacer_clustering_folder) for cas_cluster_id, group in grouped]
    logging.info(f"len(mp_args):{len(mp_args)}")

    results = []
    for args in mp_args:
        result = process_each_cas_cluster(args)
        results.append(result)

    repeat_clustering_representative_ratio_list = [res for res in results if res is not None]
    repeat_clustering_representative_ratio_frame = pd.DataFrame(repeat_clustering_representative_ratio_list,
                                                                columns=["cas_cluster_id", "main_cluster_size",
                                                                         "all_cluster_size", "ratio"])
    repeat_clustering_representative_ratio_frame.to_csv(
        f"{cas_clustering_splited_folder}cdhit_repeat_clustering_representative_ratio_part{sublist_id}.csv",
        header=True, index=None, sep="\t")

    logging.info(f"finished")


if __name__ == "__main__":
    main()