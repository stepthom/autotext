#!/bin/bash


sed -i 's/-a ./-a 1/g' tune_h1n1.sh
sbatch tune_h1n1.sh

sed -i 's/-a ./-a 2/g' tune_h1n1.sh
sbatch tune_h1n1.sh

sed -i 's/-a ./-a 4/g' tune_h1n1.sh
sbatch tune_h1n1.sh

