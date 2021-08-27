import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# import comet_ml at the top of your file
from comet_ml import Experiment
import uuid
import argparse
import numpy as np
import pandas as pd
import os

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder

import optuna

from SteveHelpers import dump_json, get_data_types, read_json
from SteveHelpers import check_dataframe, check_array
from SteveHelpers import estimate_metrics

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--project', type=str, default="eq")
    
    # Comet params
    parser.add_argument('--enable-comet', type=int, default=1)
    parser.add_argument('--comet-project-name', type=str, default="eq_searcher2")
    
    # Prep/FE params
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--pipe-args-file', type=str, default='earthquake/pipe_args_07.json')
    
    parser.add_argument('--estimator-name', type=str, default="lgbm")
    parser.add_argument('--num-cv', type=int, default=12)
    
    # Optuna params
    parser.add_argument('--optuna-study-name', type=str, default='eq_lgbm_07')
    parser.add_argument('--n-trials', type=int, default=500)

    args = parser.parse_args()
    args = vars(args)

    print("Command line arguments:")
    print(args)
    
    if args['project'] == "eq":
        from EQHelpers import run_one, get_project, get_pipeline_steps
    elif args['project'] == "h1n1":
        from H1N1Helpers import run_one, get_project, get_pipeline_steps
        
    p = get_project(args['sample_frac'])
    
    pipe_args = read_json(args["pipe_args_file"])

    # Create an experiment with your api key
    exp=None
    if args['enable_comet'] == 1:
        exp = Experiment(
            project_name=args['comet_project_name'],
            workspace="stepthom",
            parse_args=False, auto_param_logging=False, auto_metric_logging=False, log_graph=False
        )

    runname = ""
    if exp:
        runname = exp.get_key()
    else:
        runname = str(uuid.uuid4())
    print("tune_eq: Run name: {}".format(runname))

    if exp is not None:
        exp.log_other('train_fn', p.train_fn)
        exp.log_other('test_fn', p.test_fn)
        exp.log_table('X_head.csv', p.X.head())
        exp.log_parameters(args)
        exp.log_parameters(pipe_args)
        exp.log_asset('SteveHelpers.py')
        exp.log_asset('EQHelpers.py')

        
    def objective(trial, X, y, args, pipe_args, exp, runname, project):
        params = {}
        estimator = None

        if args['estimator_name'] == "lgbm":
            upper = 4096
            params = {
                
                  "max_bin": trial.suggest_int("max_bin", 32, 512),
                  
                  "num_leaves": trial.suggest_int("num_leaves", 4, upper, log=True),
                  "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.001, 128),
                  "min_child_samples": trial.suggest_int("min_child_samples", 2, 2**7, log=True),
                  "learning_rate": trial.suggest_loguniform("learning_rate", 1/256, 2.0),
                  "feature_fraction_bynode":  trial.suggest_float("feature_fraction_bynode", 0.01, 1.0),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
                  "subsample": 1.0,
                  "subsample_freq": 0,
                  "reg_alpha": trial.suggest_loguniform("reg_alpha", 1/1024, 1024),
                  "reg_lambda": trial.suggest_loguniform("reg_lambda", 1/1024, 1024),
                  "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),

                  "n_estimators": 10000,
                  "boosting_type": 'gbdt',
                  "max_depth": -1,
                  "n_jobs": 5,
                  #"objective": project.objective,
                  #"num_class": project.num_class,
                  "verbosity": -1,
                  "seed": 77,
            }

            estimator = LGBMClassifier(**params)
            
        elif args['estimator_name'] == "xgboost":
            upper = 4096
            params = {
                  "max_leaves": trial.suggest_int("max_leaves", 4, upper, log=True),
                  "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.001, 128),
                  "learning_rate": trial.suggest_loguniform("learning_rate", 1/1024, 1.0),
                  "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                  "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1.0),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
                  "reg_alpha": trial.suggest_loguniform("reg_alpha", 1/1024, 1024),
                  "reg_lambda": trial.suggest_loguniform("reg_lambda", 1/1024, 1024),
                  "gamma": trial.suggest_loguniform("gamma", 1/1024, 128),

                  "n_estimators": 3500,
                  "booster": "gbtree",
                  "max_depth": 0,
                  "grow_policy": "lossguide",
                  "tree_method": "hist",
                  "n_jobs": 5,
                  #"objective": "multi:softproj" if project.objective == "multiclass" else "binary",
                  "eval_metric": "mlogloss",
                  "verbosity": 1,
                  #"num_class": 3,
                  "num_class": project.num_class,
                  "seed": 77,
            }
            estimator = XGBClassifier(**params)
        elif args['estimator_name'] == "rf":
            upper = 4096
            params = {
                  "n_estimators": trial.suggest_int("n_estimators", 64, upper, log=True),
                  "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 64, upper, log=True),
                  "max_features": trial.suggest_float("max_features", 0.1, 1.0),
                  "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy']),
                  "class_weight": trial.suggest_categorical("class_weight", ['balanced', None]),
                  "ccp_alpha": trial.suggest_loguniform("ccp_alpha", 1/1024, 0.1),
                
                  "random_state": 77,
                  "n_jobs": 5,
            }
            estimator = RandomForestClassifier(**params)
            
        elif args['estimator_name'] == "lr":
            params = {
                  "penalty": trial.suggest_categorical("penalty", ['l1', 'l2', 'elasticnet', 'none']),
                  "C": trial.suggest_float("C", 0.7, 1.3),
                  "class_weight": trial.suggest_categorical("class_weight", ['balanced', None]),
                  "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
               
                  "solver": "saga",
                  "random_state": 77,
                  "n_jobs": 5,
                  "max_iter": 500,
            }
            estimator = LogisticRegression(**params)
            
        elif args['estimator_name'] == "hist":
            upper = 400
            params = {
                  "max_iter": trial.suggest_int("max_iter", 32, upper, log=True),
                  "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 32, upper*2, log=True),
                  "learning_rate": trial.suggest_loguniform("learning_rate", 0.05, 1.0),
                  "l2_regularization": trial.suggest_loguniform("l2_regularization", 1/1024, 0.1),
               
                  "random_state": 77,
            }
            estimator = HistGradientBoostingClassifier(**params)
        else:
            print("Unknown estimator name {}".format(estimator_name))

        metrics = estimate_metrics(p.X, p.y, pipe_args, get_pipeline_steps, estimator, args['num_cv'], early_stop=True, project=p)

        if exp is not None:
            exp.log_metric("mean_val_score", np.mean(metrics['val_scores']), step=trial.number)
            exp.log_metric("mean_train_score", np.mean(metrics['train_scores']), step=trial.number)
            exp.log_text(params, step=trial.number)

        # Log for later
        res = {}
        res['runname'] = runname
        res['trial'] = trial.number
        res['args'] = args
        res['pipe_args'] = pipe_args
        res['params'] = params
        res['metrics'] = metrics
        json_fn = os.path.join(p.out_dir, "{}-trial-{}.json".format(runname, trial.number))
        dump_json(json_fn, res)

        return np.mean(metrics['val_scores'])
    
    study = optuna.create_study(
        study_name=args['optuna_study_name'],
        storage="sqlite:///eq_studies.db",
        sampler= optuna.samplers.TPESampler(
            n_startup_trials = 100,
            n_ei_candidates = 10,
            constant_liar=True,
        ),
        direction="maximize",
        load_if_exists = True,
    )
    
    study.optimize(lambda trial: objective(trial, p.X, p.y, args, pipe_args, exp, runname, p),
                    n_trials=args['n_trials'], 
                    gc_after_trial=True)

    return
    
if __name__ == "__main__":
    main()
