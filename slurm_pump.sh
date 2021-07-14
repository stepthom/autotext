#!/bin/bash
#SBATCH --job-name=PumpSearch            
#SBATCH --cpus-per-task=10   
#SBATCH --mem=160gb
#SBATCH --time=35:00:00      

# commands for your job go here
pwd
which python
source flaml_env_slurm/bin/activate
which python
python run_lots.py --num-pumps 10
