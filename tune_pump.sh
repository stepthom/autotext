#!/bin/bash
#SBATCH --job-name=TunePump
#SBATCH --cpus-per-task=10   
#SBATCH --mem=50gb
#SBATCH --time=30:00:00      

# commands for your job go here
pwd
which python
source flaml_env_slurm/bin/activate
which python
OS_STEVE_MIN_SAMPLE_LEAF=25 OS_STEVE_SMOOTHING=50.0 python tune_pump.py -g 3 -a 2
