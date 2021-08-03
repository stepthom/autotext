#!/bin/bash


declare -a list1=("1" "3" "5" )
declare -a list2=("0.1" "0.5" "1.0" )

for str1 in "${list1[@]}"
do
    for str2 in "${list2[@]}"
    do

        sed -i  "s|OS_STEVE_MIN_SAMPLE_LEAF=\S*|OS_STEVE_MIN_SAMPLE_LEAF=${str1}|g; s|OS_STEVE_SMOOTHING=\S*|OS_STEVE_SMOOTHING=${str2}|g; s|-g .|-g 3|g; s|-a .|-a 3|g" tune_eq.sh
        sbatch tune_eq.sh

    done
done

