#!/bin/bash



sed -i 's/-a ./-a 1/g' tune_seasonal.sh
sbatch tune_seasonal.sh


sed -i 's/-a ./-a 2/g' tune_seasonal.sh
sbatch tune_seasonal.sh

sed -i 's/-a ./-a 4/g' tune_seasonal.sh
sbatch tune_seasonal.sh

