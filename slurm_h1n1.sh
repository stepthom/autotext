#!/bin/bash
#SBATCH --job-name=H1N1Search            
#SBATCH --cpus-per-task=10   
#SBATCH --mem=260gb
#SBATCH --time=50:00:00      

# commands for your job go here
pwd
which python
source flaml_env_slurm/bin/activate
which python
python run_parallel.py --run-settings full_settings.json --num-h1n1s 10
