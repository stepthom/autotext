import argparse
import subprocess
from itertools import product

# Hyperparam tuning using slurm.
# For each hyper param combo, generate a temporary bash script with the appropriate call to tune.
# Then, call sbatch with the temporary bash script.

def main():
  
    # xgboost
    #boosters = ['gbtree']
    
    # lgbm
    boosters = ['gbdt', 'rf', "dart"]
    #grow_policys = ['depthwise', 'lossguide']
    grow_policys =  ["lossguide"]
    scale_pos_weights = [1.0]
    samplers = ["tpe", "tpe", "random", "motpe"]
    geo_id_sets = [2, 3]
    
    hours = 96

    all_combos = list(product(boosters,
                              grow_policys,
                              scale_pos_weights,
                              samplers,
                              geo_id_sets,
                             ))

    print("Number of combos: {}".format(len(all_combos)))
    for combo in all_combos:

        booster = combo[0]
        grow_policy = combo[1]
        scale_pos_weight = combo[2]
        sampler = combo[3]
        geo_id_set = combo[4]

        string = '\n'.join([
            "#!/bin/bash",
            "#SBATCH --job-name=OptunaEQ",
            "#SBATCH --cpus-per-task=10"  ,
            "#SBATCH --mem={}gb".format(80),
            "#SBATCH --time={}:00:00".format(hours+1),
            "#SBATCH --output=slurm/R-%x-%j.out",

            "",
            "pwd",
            "which python",
            "source flaml_env_slurm/bin/activate",
            "which python",
            "python tune_eq_dev.py --algo-set 1 --n-trials 5000 --run-type optuna --geo-id-set {} --cat-encoder 2 --booster {} --grow-policy {} --scale-pos-weight {} --sampler {}".format(
                geo_id_set,
                booster,
                grow_policy,
                scale_pos_weight,
                sampler,
            ),
        ]
        )

        f_name = 'tmp_scripts/tune_eq_optuna_tmp.sh'
        with open(f_name, "w") as text_file:
            text_file.write(string)

        cmd = [ "sbatch", f_name ]
        p = subprocess.Popen(cmd, universal_newlines=True, bufsize=1)
        exit_code = p.wait()


if __name__ == "__main__":
    main()
