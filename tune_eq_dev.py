import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# import comet_ml at the top of your file
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
from SteveHelpers import StudyData

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--study-name', type=str, default='h1n1')
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--num-cv', type=int, default=12)
    
    # Optuna params
    parser.add_argument('--n-trials', type=int, default=500)

    args = parser.parse_args()
    args = vars(args)
    
    print("Command line arguments:")
    print(args)
        
    def objective(trial, study_name, project):
        estimator_name = trial.suggest_categorical("estimator_name", ["lgbm", "xgboost", "rf", "hist"])
        
        if study_name == "h1n1":
            pipe_name = trial.suggest_categorical("pipe_name", ['01', '02'])
        elif study_name == "seasonal":
            pipe_name = trial.suggest_categorical("pipe_name", ['01', '02'])
        elif study_name == "eq":
            pipe_name = trial.suggest_categorical("pipe_name", ['01', '02', '03', '04'])
        elif study_name == "pump":
            pipe_name = trial.suggest_categorical("pipe_name", ['01', '02', '03', '04'])
    
        estimator = None
        params = {}
        if estimator_name == "lgbm":
            upper = 4096
            params = {
                  "max_bin": trial.suggest_int("max_bin", 32, 512),
                  "num_leaves": trial.suggest_int("num_leaves", 4, upper, log=True),
                  "min_child_weight": trial.suggest_loguniform("lgbm_min_child_weight", 0.001, 128),
                  "min_child_samples": trial.suggest_int("min_child_samples", 2, 2**7, log=True),
                  "learning_rate": trial.suggest_loguniform("lgbm_learning_rate", 1/256, 2.0),
                  "feature_fraction_bynode":  trial.suggest_float("feature_fraction_bynode", 0.01, 1.0),
                  "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.01, 1.0),
                  "subsample": 1.0,
                  "subsample_freq": 0,
                  "reg_alpha": trial.suggest_loguniform("lgbm_reg_alpha", 1/1024, 1024),
                  "reg_lambda": trial.suggest_loguniform("lgbm_reg_lambda", 1/1024, 1024),
                  "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),

                  "n_estimators": 5000,
                  "boosting_type": 'gbdt',
                  "max_depth": -1,
                  "n_jobs": 5,
                  #"objective": project.objective,
                  #"num_class": project.num_class,
                  "verbosity": -1,
                  "seed": 77,
            }
            estimator = LGBMClassifier(**params)
            
        elif estimator_name == "xgboost":
            upper = 4096
            params = {
                  "max_leaves": trial.suggest_int("xgboost_max_leaves", 4, upper, log=True),
                  "min_child_weight": trial.suggest_loguniform("xgboost_min_child_weight", 0.001, 128),
                  "learning_rate": trial.suggest_loguniform("xgboost_learning_rate", 1/1024, 1.0),
                  "subsample": trial.suggest_float("xgboost_subsample", 0.1, 1.0),
                  "colsample_bylevel": trial.suggest_float("xgboost_colsample_bylevel", 0.01, 1.0),
                  "colsample_bytree": trial.suggest_float("xgboost_colsample_bytree", 0.01, 1.0),
                  "reg_alpha": trial.suggest_loguniform("xgboost_reg_alpha", 1/1024, 1024),
                  "reg_lambda": trial.suggest_loguniform("xgboost_reg_lambda", 1/1024, 1024),
                  "gamma": trial.suggest_loguniform("gamma", 1/1024, 128),

                  "n_estimators": 5000,
                  "booster": "gbtree",
                  "max_depth": 0,
                  "grow_policy": "lossguide",
                  "tree_method": "hist",
                  "n_jobs": 5,
                  #"objective": "multi:softproj" if project.objective == "multiclass" else "binary",
                  #"eval_metric": "mlogloss",
                  "verbosity": 1,
                  #"num_class": 3,
                  #"num_class": project.num_class,
                  "seed": 77,
            }
            estimator = XGBClassifier(**params)
            
        elif estimator_name == "rf":
            upper = 4096
            params = {
                  "n_estimators": trial.suggest_int("rf_n_estimators", 64, upper, log=True),
                  "max_leaf_nodes": trial.suggest_int("rf_max_leaf_nodes", 64, upper, log=True),
                  "max_features": trial.suggest_float("rf_max_features", 0.1, 1.0),
                  "criterion": trial.suggest_categorical("criterion", ['gini', 'entropy']),
                  "class_weight": trial.suggest_categorical("rf_class_weight", ['balanced', None]),
                  "ccp_alpha": trial.suggest_loguniform("ccp_alpha", 1/1024, 0.1),
                
                  "random_state": 77,
                  "n_jobs": 5,
            }
            estimator = RandomForestClassifier(**params)
            
        elif estimator_name == "lr":
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
            
        elif estimator_name == "hist":
            upper = 400
            params = {
                  "max_iter": trial.suggest_int("hist_max_iter", 32, upper, log=True),
                  "max_leaf_nodes": trial.suggest_int("hist_max_leaf_nodes", 32, upper*2, log=True),
                  "learning_rate": trial.suggest_loguniform("hist_learning_rate", 0.05, 1.0),
                  "l2_regularization": trial.suggest_loguniform("hist_l2_regularization", 1/1024, 0.1),
               
                  "random_state": 77,
            }
            estimator = HistGradientBoostingClassifier(**params)
        else:
            print("Unknown estimator name {}".format(estimator_name))

        metrics = estimate_metrics(study_data, study_name, pipe_name, estimator, args['num_cv'], early_stop=True, metric=study.user_attrs['metric'])
            
        # Log for later
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("estimator_name", estimator_name)
        trial.set_user_attr("estimator_params", params)

        return np.mean(metrics['val_scores'])
    
    study = optuna.load_study(
        study_name=args['study_name'],
        storage="postgresql://hpc3552@172.20.13.14/hpc3552",
    )
   
    study_data = StudyData(study, args['sample_frac'])
    
    study.optimize(lambda trial: objective(trial, args['study_name'], study_data),
                    n_trials=args['n_trials'], 
                    gc_after_trial=True)

    return
    
if __name__ == "__main__":
    main()
