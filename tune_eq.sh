#!/bin/bash
#SBATCH --job-name=TuneEQ
#SBATCH --cpus-per-task=10   
#SBATCH --mem=50gb
#SBATCH --time=20:00:00      
#SBATCH --output=slurm/R-%x-%j.out

# commands for your job go here
pwd
which python
source flaml_env_slurm/bin/activate
which python
OS_STEVE_MIN_SAMPLE_LEAF=5 OS_STEVE_SMOOTHING=1.0 python tune_eq.py -g 3 -a 3
