Directory structure for each comp should be like this (e.g., for `pump` comp):

```
pump # Holds raw data
pump/data # Holds preprocessed train and test files
pump/out  # Holds results of training runs (i.e., JSON log, and prediction/probas files)
```

Everything is based on Optuna's `study` and `trial` concepts as much as possible.

Workflow:

- Setup
  - Download, configure, make, and make install local build of PG. 
  - Install packages from requirements.txt
-  Start the PG server
  - `bin/postgres -D ./optuna_study_dir`
  - Create a database.
- Create the studies with `create_studies.py`
- Level 1
  - Tune the studies via `tune_eq_dev.py` as much as you'd like.
    - Can run multiple at the same time; on slurm (see `slurm_scripts`); etc.
    - Tuning includes pipeline, algo, algo hyper params
  - View results with `get_study_scores.py`
- Level 2
  - Select one or more trial numbers that you want to run full
  - Run them `level2_eq.dev` to create preds and probas files
  - Soft vote whatever you like via `soft_voting.py`
- Submit and win


Python's `venv` is useful for managing packages. All required packages for FLAML are in `flaml_env` or `flaml_env_slurm`
