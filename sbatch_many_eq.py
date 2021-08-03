import argparse
import subprocess

# Hyperparam tuning using slurm.
# For each hyper param combo, generate a temporary bash script with the appropriate call to tune.
# Then, call sbatch with the temporary bash script.

def main():    
   
    for geo_id_set in [2, 3]:
        for algo_set in [1, 3]:
            for min_sample_leaf in [1, 3, 5]:
                for smoothing in [0.1, 0.5, 1.0]:
                    for n_hidden in [2, 4, 8, 16]:
                        for smooth_marginals in ["true", "false"]:

                            string = '\n'.join([
                                "#!/bin/bash",
                                "#SBATCH --job-name=TuneEQ",
                                "#SBATCH --cpus-per-task=10"  ,
                                "#SBATCH --mem=20gb",
                                "#SBATCH --time=20:00:00",
                                "#SBATCH --output=slurm/R-%x-%j.out",

                                "",
                                "pwd",
                                "which python",
                                "source flaml_env_slurm/bin/activate",
                                "which python",
                                "python tune_eq.py --geo-id-set {} --algo-set {} --min-sample-leaf {} --smoothing {} --n_hidden {} --smooth-marginals {}".format(
                                    geo_id_set, algo_set, min_sample_leaf, smoothing, n_hidden, smooth_marginals),
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
