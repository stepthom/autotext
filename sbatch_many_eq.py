import argparse
import subprocess
from itertools import product

# Hyperparam tuning using slurm.
# For each hyper param combo, generate a temporary bash script with the appropriate call to tune.
# Then, call sbatch with the temporary bash script.

def main():

    geo_id_sets = [3, 2]
    algo_sets = [1, 2, 3]
    min_sample_leafs = [1, 3, 5]
    smoothings = [0.1, 1.0, 10]
    n_hiddens = [6]
    smooth_marginals = [0]

    autofeats = [0, 1]

    #normalizes = ["true", "false"]
    normalizes = [0, 1]

    ensembles = [0]
    #ensembles = ["true"]

    all_combos = list(product(geo_id_sets, algo_sets,
                              min_sample_leafs, smoothings,
                              n_hiddens, smooth_marginals,
                              autofeats, normalizes,
                              ensembles))

    print("Number of combos: {}".format(len(all_combos)))
    for combo in all_combos:

        geo_id_set = combo[0]
        algo_set = combo[1]
        min_sample_leaf = combo[2]
        smoothing = combo[3]
        n_hidden = combo[4]
        smooth_marginals = combo[5]
        autofeat = combo[6]
        normalize = combo[7]
        ensemble = combo[8]

        string = '\n'.join([
            "#!/bin/bash",
            "#SBATCH --job-name=TuneEQ",
            "#SBATCH --cpus-per-task=10"  ,
            "#SBATCH --mem=30gb",
            "#SBATCH --time=20:00:00",
            "#SBATCH --output=slurm/R-%x-%j.out",

            "",
            "pwd",
            "which python",
            "source flaml_env_slurm/bin/activate",
            "which python",
            "python tune_eq_dev.py --time-budget 50000 --geo-id-set {} --algo-set {} --min-sample-leaf {} --smoothing {} --n-hidden {} --smooth-marginals {} --autofeat {} --normalize {} --ensemble {}".format(
                geo_id_set, algo_set, min_sample_leaf, smoothing, n_hidden, smooth_marginals, autofeat, normalize, ensemble),
        ]
        )

        f_name = 'tmp_scripts/tune_eq_tmp.sh'
        with open(f_name, "w") as text_file:
            text_file.write(string)

        cmd = [ "sbatch", f_name ]
        p = subprocess.Popen(cmd, universal_newlines=True, bufsize=1)
        exit_code = p.wait()


if __name__ == "__main__":
    main()
