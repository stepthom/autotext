Directory structure for each comp should be like this (e.g., for `pump` comp):

```
pump # Holds raw data
pump/data # Holds preprocessed train and test files
pump/out  # Holds results of training runs (i.e., JSON log, and prediction/probas files)
```

`prep_gen.py` is a general preprocessor for all comps.  When it generates a train/test file, it will also create a "data sheet" that describes that train/test file, i.e., what preprocessing steps were used, and given it a uniuqe data id.

`run_one.py` will run a single FLAML search on a single train/test file. It is smart enough to find a data_sheet to run at random, and will not re-run on the same data sheet with the same settings. A run will create a unique runname. Output will be saved to `out`

`run_seq.py` will call `run_one.py` as a subprocess N times in a row, sequentially.

`run_parallel.py` will call N `run_seq.py` in parallel and wait.

`slurm_*.sh` will submit a `run_parallel.py` to tthe Frontenac compute cluster.

Python's `venv` is useful for manageing packages. All required packages for FLAML are in `flaml_env` or `flaml_env_slurm`
