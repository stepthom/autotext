#!/bin/bash
#SBATCH --job-name=PumpSearch            
#SBATCH --cpus-per-task=10   
#SBATCH --mem=250gb
#SBATCH --time=50:00:00      

# commands for your job go here
pwd
which python
source flaml_env_slurm/bin/activate
which python
python run_parallel.py --run-settings full_settings.json --num-pumps 10
