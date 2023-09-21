#!/bin/bash
# https://docs.rc.fas.harvard.edu/kb/running-jobs/#articleTOC_8

#SBATCH -c 2                # Number of cores (-c)
#SBATCH -t 0-00:10          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p test             # Partition to submit to
#SBATCH --mem=16000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o /n/home01/rspang/results/job_stdout_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -o /n/home01/rspang/results/job_errout_%j.err  # File to which STDERR will be written, %j inserts jobid

# load modules
module load Mambaforge/23.3.1-fasrc01

# set python environmant
mamba activate workshop

# run code
python global_precipitation_sentiment.py
