#!/bin/bash
#SBATCH --job-name=SeasonalSearch            
#SBATCH --cpus-per-task=10   
#SBATCH --mem=260gb
#SBATCH --time=35:00:00      

# commands for your job go here
pwd
which python
source flaml_env_slurm/bin/activate
which python
python run_lots.py --num-earthquakes 10