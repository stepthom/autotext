#!/bin/bash
#SBATCH --job-name=TuneH1N1
#SBATCH --cpus-per-task=10   
#SBATCH --mem=50gb
#SBATCH --time=20:00:00      

# commands for your job go here
pwd
which python
source flaml_env_slurm/bin/activate
which python
OS_STEVE_MIN_SAMPLE_LEAF=25 OS_STEVE_SMOOTHING=50.0 python tune_h1n1.py -a 3
