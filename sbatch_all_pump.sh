#!/bin/bash



sed -i 's/-g ./-g 1/g; s/-a ./-a 1/g' tune_pump.sh
sbatch tune_pump.sh

sed -i 's/-g ./-g 2/g; s/-a ./-a 1/g' tune_pump.sh
sbatch tune_pump.sh

sed -i 's/-g ./-g 3/g; s/-a ./-a 1/g' tune_pump.sh
sbatch tune_pump.sh

sed -i 's/-g ./-g 1/g; s/-a ./-a 2/g' tune_pump.sh
sbatch tune_pump.sh

sed -i 's/-g ./-g 2/g; s/-a ./-a 2/g' tune_pump.sh
sbatch tune_pump.sh

sed -i 's/-g ./-g 3/g; s/-a ./-a 2/g' tune_pump.sh
sbatch tune_pump.sh

sed -i 's/-g ./-g 1/g; s/-a ./-a 4/g' tune_pump.sh
sbatch tune_pump.sh

sed -i 's/-g ./-g 2/g; s/-a ./-a 4/g' tune_pump.sh
sbatch tune_pump.sh

sed -i 's/-g ./-g 3/g; s/-a ./-a 4/g' tune_pump.sh
sbatch tune_pump.sh

