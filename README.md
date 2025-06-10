# CRISPR-PAMdb
## Steps of the Mining Pipeline
### Identifying CRISPR Repeats
In the first step we detect CRISPR repeats using [PILER-CR](https://doi.org/10.1186/1471-2105-8-18) and 
[MinCED](https://github.com/ctSkennerton/minced). We then identify the contigs containing the repeats and merge the 
results of the two tools. 

### Identify Cas Proteins and CRISPR Array

On the contigs, we search for protein coding genes using [prodigal](https://doi.org/10.1186/1471-2105-11-119). Using
[hmmsearch](https://doi.org/10.1093/nar/gkr367), we identify Cas proteins among the protein coding genes. The found Cas 
proteins are then checked if they are in flanking regions of the CRISPR repeats.

### Cluster Cas Proteins
To enhance the number of spacers associated with each Cas ortholog, we pooled CRISPR arrays from
closely related Cas proteins. Proteins are clustered using MMseqs2 at 98% amino acid identity.

### Identify PAM and Protospacers
We then identify PAM sequences and protospacers based on the identified CRISPR-Cas regions.

## How to run the Mining Pipeline
### Cloning the Git Repository

In order to run the pipeline, you need to clone the git repository. This is most easily done by using 
[Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git):
```sh
git clone https://github.com/Schwank-Lab/CRISPR-PAMdb.git
```
If you don't want to install or use Git, you can also download the repository on the 
[repository website](https://github.com/Schwank-Lab/CRISPR-PAMdb.git). Click on the button `Code` and then 
`Download Zip`. After the download finished, unzip the folder.

### Setting up the Conda Environment
In order to run the pipeline, you need to have 
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.
Create the environment using the provided yaml file. Adapt the path to the location of the repository.
```sh
conda env create -f ~/path/to/CRISPR-PAMdb/snakemake_pipeline/envs/environment.yml
```
Please install PAMpredict according to [their instructions](https://github.com/Matteo-Ciciani/PAMpredict).

### Snakemake Pipeline
The snakemake pipeline expects gzipped files as input and is written to use the SLURM queueing system. Set your 
preferences in the config and add the relative path from the `input_directory` to the gzipped fasta files to the 
samples file. You can the pipeline run it with:
```sh
conda activate mining
snakemake -s /path/to/CRISPR-PAMdb/snakemake_pipeline/Snakefile --configfile /path/to/CRISPR-PAMdb/snakemake_pipeline/config/config_template.yaml -j 1 --cluster-cancel scancel --use-conda --dryrun
```
Check if the steps listed in the snakemake dry run are what you are planning to run. If this is correct, run:
```sh
snakemake -s /path/to/CRISPR-PAMdb/snakemake_pipeline/Snakefile --configfile /path/to/CRISPR-PAMdb/snakemake_pipeline/config/config_template.yaml -j 1 --cluster-cancel scancel --use-conda
```

