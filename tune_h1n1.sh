#!/bin/bash
#SBATCH --job-name=TuneH1N1
#SBATCH --cpus-per-task=10   
#SBATCH --mem=20gb
#SBATCH --time=30:00:00      

# commands for your job go here
pwd
which python
source flaml_env_slurm/bin/activate
which python
python tune_h1n1.py -a 4
