import os
import subprocess
from Bio import SeqIO
import logging
import argparse
import glob

##################
# run PAMpredict #
##################
def run_pam_predict(spacer_file, pam_predict_installation_directory,
                    phage_blastn_database_folder,
                    pam_prediction_output_directory):
    """
    Runs the PAMpredict tool.
    """
    further_output_folder = f"{pam_prediction_output_directory}"
    if not os.path.exists(further_output_folder):
        os.makedirs(further_output_folder)

    # Check if the file has more than 10 sequences
    seq_count = sum(1 for _ in SeqIO.parse(spacer_file, "fasta"))
    if seq_count <= 10:
        logging.info(f"Skipping {spacer_file} as it has {seq_count} sequences (<= 10).")
        return

    # Define the command and arguments as a list
    skw = dict(stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    command = [
        f"{pam_predict_installation_directory}/PAMpredict.py",
        "--keep_tmp",  #: Keep temporary files
        "--force",  #: Overwrites existing results if present.
        "--threads", "1",  #: Number of threads to use. it will just use more cpu when use more threads
        spacer_file,
        f"{phage_blastn_database_folder}",
        f"{pam_prediction_output_directory}"
    ]
    result = subprocess.run(command, universal_newlines = True, **skw)
    if result.returncode != 0:
        print("Command failed with return code:", result.returncode)
        print("Command:", " ".join(command))
        raise subprocess.CalledProcessError(result.returncode, command, result.stderr)

def process_each_cas_cluster(spacer_file, pam_predict_installation_directory,
                             phage_database_list, pam_prediction_output_directory):  # define function here so all variables in the main function is accessible
    """
    Process a single cas cluster. Encapsulates all steps for processing one group.
    """
    try:
        # Run PAMpredict
        for phage_folder, _ in phage_database_list:
            run_pam_predict(spacer_file=spacer_file, pam_predict_installation_directory=pam_predict_installation_directory,
                            phage_blastn_database_folder=phage_folder, pam_prediction_output_directory=pam_prediction_output_directory)
    except Exception as e:
        print(f"Error processing {spacer_file}: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PAM Identification Script")
    parser.add_argument("--output_directory", type = str, required = True, help = "Output folder path", dest = "output_directory")
    parser.add_argument("--cas_type", type = str, default = "cas9", help = "CAS type (default: cas9)", dest = "cas_type")
    parser.add_argument("--pam_predict_installation_directory", type = str, help = "PAMpredict installation folder path", dest = "pam_predict_installation_directory")
    parser.add_argument("--pam_prediction_output_directory", type = str, help = "PAMpredict output folder path", dest = "pam_prediction_output_directory")
    parser.add_argument("--phage_database", type = str, help = "Phage database folder path", dest = "phage_database")
    parser.add_argument("--fasta", type = str, help = "spacers", dest = "fasta")

    args = parser.parse_args()
    output_directory = args.output_directory
    cas_type = args.cas_type
    pam_predict_installation_directory = args.pam_predict_installation_directory
    pam_prediction_output_directory = args.pam_prediction_output_directory
    phage_database = args.phage_database
    fasta = args.fasta
    fastas = glob.glob(os.path.join(fasta,'*.fasta'))

    all_cas_proteins_folder = f"{output_directory}/{cas_type}/all_{cas_type}_proteins/"
    cas_clustering_folder = f"{all_cas_proteins_folder}mmseqs2_clustering/"
    cdhit_repeat_oriented_spacer_clustering_folder = f"{cas_clustering_folder}cdhit_repeat_oriented_spacer_clustering/"

    phage_database_list = [
        (os.path.dirname(phage_database),
         phage_database)
    ]

    phage_database_name = os.path.basename(phage_database_list[0][0])
    further_output_folder = f"{pam_prediction_output_directory}/{phage_database_name}/"

    logging.info(f"start")

    for fasta_file in fastas:
        pam_prediction_output_directory = os.path.join(further_output_folder, os.path.basename(fasta_file).replace('.fasta', ''))
        process_each_cas_cluster(fasta_file, pam_predict_installation_directory, phage_database_list, pam_prediction_output_directory)

    logging.info(f"finished")
