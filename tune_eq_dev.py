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
import sys

import scipy.stats
import json
from json.decoder import JSONDecodeError
import socket
import datetime

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector

import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper

from autofeat import AutoFeatRegressor, AutoFeatClassifier, AutoFeatLight

import optuna
import time

from functools import partial
import json

from SteveHelpers import dump_json, get_data_types
from SteveHelpers import check_dataframe, check_array
from SteveHelpers import get_pipeline_steps

def estimate_metrics(X, y, pipe_args, estimator, num_cv=5):
    """
    X, y: features and target
    args: args to control FE pipeline
    estimator
    
    Returns metrics, such as:
    val_scores: estimated val score for each each CV fold
    train_scores: estimated val score for each each CV fold
    """
    metrics = {}
    val_scores = []
    train_scores = []
    train_times = []
    best_iterations = []
    
    print("estimate_metrics: pipe_args: {}".format(pipe_args))
          
    skf = StratifiedKFold(n_splits=num_cv, random_state=42, shuffle=True)

    cv_step = -1
    for train_index, val_index in skf.split(X, y):
        cv_step = cv_step + 1
        print("estimate_metrics: cv_step {} of {}".format(cv_step, num_cv))

        X_train, X_val = X.loc[train_index].reset_index(drop=True), X.loc[val_index].reset_index(drop=True)
        y_train, y_val = y[train_index], y[val_index]

        steps = get_pipeline_steps(pipe_args)
        pipe = Pipeline(steps)

        pipe.fit(X_train, y_train)

        _X_train = pipe.transform(X_train)
        _X_val  = pipe.transform(X_val)

        check_dataframe(_X_train, "_X_train")
        check_dataframe(_X_val, "_X_val")
        
        extra_fit_params = {}
        if isinstance (estimator, LGBMClassifier) or isinstance(estimator, XGBClassifier):
            extra_fit_params.update({
                'eval_set': [(_X_val, y_val)],
                'early_stopping_rounds':20,
                'verbose': 50,
            })
        if isinstance (estimator, LGBMClassifier) or isinstance(estimator, HistGradientBoostingClassifier) or isinstance(estimator, XGBClassifier):
            indices =  [i for i, ix in enumerate(_X_train.columns.values) if "_oenc" in ix]
            if len(indices) > 0:
                if isinstance (estimator, LGBMClassifier):
                    extra_fit_params.update({
                        'categorical_feature': indices, 
                    })
                elif isinstance (estimator, HistGradientBoostingClassifier):
                    estimator.set_params(**{
                        'categorical_features': indices, 
                    })
                elif isinstance (estimator, XGBClassifier):
                    # This appears to not be working; xgboost complains.
                    # Bummber!
                    #estimator.set_params(**{
                        #'enable_categorical': True, 
                    #})
                    pass
        print("estimate_metric: extra_fit_params:")
        print(extra_fit_params)

        start = time.time()
        print("estimate_metrics: estimator: {}".format(estimator))
        print("estimate_metric: fitting...: ")
        estimator.fit(_X_train, y_train, **extra_fit_params)
        train_times.append((time.time() - start))

        print("estimate_metric: Val: ")
        y_val_pred_proba = estimator.predict_proba(_X_val)
        y_val_pred = estimator.predict(_X_val)
        val_scores.append(f1_score(y_val, y_val_pred, average="micro"))
        print(classification_report(y_val, y_val_pred, digits=4))

        print("estimate_metric: Train: ")
        y_train_pred_proba = estimator.predict_proba(_X_train)
        y_train_pred = estimator.predict(_X_train)
        train_scores.append(f1_score(y_train, y_train_pred, average="micro"))
        print(classification_report(y_train, y_train_pred, digits=4))

        bi = None
        if hasattr(estimator, 'best_iteration_'):
            bi = estimator.best_iteration_
        elif hasattr(estimator, 'best_iteration'):
            bi = estimator.best_iteration
        if bi is not None:
            best_iterations.append(bi)
          
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

    metrics['val_scores'] = val_scores
    metrics['val_scores_range'] = mean_confidence_interval(val_scores)
    metrics['train_scores'] = train_scores
    bir = []
    if len(best_iterations) > 0:
        bir = mean_confidence_interval(best_iterations)
    metrics['best_iterations_range'] = bir
    metrics['train_times'] = train_times
    
    print("estimate_metrics: cv complete.")
    print("estimate_metrics: metrics:")
    print(metrics)
          
    return metrics


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--id-col', type=str, default="building_id")
    parser.add_argument('--target-col', type=str, default="damage_grade")
    parser.add_argument('--out-dir', type=str, default="earthquake/out")
    
    # Comet params
    parser.add_argument('--enable-comet', type=int, default=1)
    
    # Prep/FE params
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--pipe-args-file', type=str, default='earthquake/pipe_args_01.json')
    
    parser.add_argument('--estimator-name', type=str, default="lgbm")
    parser.add_argument('--num-cv', type=int, default=12)
    
    # Optuna params
    parser.add_argument('--sampler', type=str, default='tpe')
    parser.add_argument('--n-trials', type=int, default=3)

    args = parser.parse_args()
    args = vars(args)

    print("Command line arguments:")
    print(args)

    data_id = '000'

    # Create an experiment with your api key
    exp=None
    if args['enable_comet'] == 1:
        exp = Experiment(
            project_name="eq_searcher2",
            workspace="stepthom",
            parse_args=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            log_graph=False
        )

    train_fn = 'earthquake/earthquake_train.csv'
    test_fn = 'earthquake/earthquake_test.csv'
    train_df = pd.read_csv(train_fn)
    test_df = pd.read_csv(test_fn)
    if args['sample_frac']< 1.0:
        train_df  = train_df.sample(frac=args['sample_frac'], random_state=3).reset_index(drop=True)
        
    X = train_df.drop([args['id_col'], args['target_col']], axis=1)
    y = train_df[args['target_col']]
    X_test = test_df.drop([args['id_col']], axis=1)
    
    label_transformer = LabelEncoder()
    y = label_transformer.fit_transform(y)
                 
    pipe_args = {}
    with open(args["pipe_args_file"]) as f:
        try:
            pipe_args = json.load(f)
        except JSONDecodeError as e:
            print("ERROR: cannot parse json file {}".format(args.pipe_arg_file))
            print(e)

    runname = ""
    if exp:
        runname = exp.get_key()
    else:
        runname = str(uuid.uuid4())
    print("tune_eq: Run name: {}".format(runname))

    starttime = datetime.datetime.now()
    
    if exp is not None:
        exp.log_other('train_fn', train_fn)
        exp.log_other('test_fn', test_fn)
        exp.log_table('X_head.csv', X.head())
        exp.log_parameters(args)
        exp.log_parameters(pipe_args)
        exp.log_asset('SteveHelpers.py')

        
    def objective(trial, X, y, args, pipe_args, exp, runname):
        params = {}
        estimator = None

        if args['estimator_name'] == "lgbm":
            upper = 4096
            params = {
                  "num_leaves": trial.suggest_int("num_leaves", 4, upper, log=True),
                  "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.001, 128),
                  "min_child_samples": trial.suggest_int("min_child_samples", 2, 2**7, log=True),
                  "learning_rate": trial.suggest_loguniform("learning_rate", 1/1024, 1.0),
                  "feature_fraction_bynode":  trial.suggest_float("feature_fraction_bynode", 0.01, 1.0),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
                  "subsample": 1.0,
                  "subsample_freq": 0,
                  "reg_alpha": trial.suggest_loguniform("reg_alpha", 1/1024, 1024),
                  "reg_lambda": trial.suggest_loguniform("reg_lambda", 1/1024, 1024),
                  "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),

                  "n_estimators": 3500,
                  "boosting_type": 'gbdt',
                  "max_depth": -1,
                  "n_jobs": 5,
                  "objective": "multiclass",
                  "verbosity": -1,
                  "num_class": 3,
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
                  "objective": "multi:softprob",
                  "eval_metric": "mlogloss",
                  "verbosity": 1,
                  "num_class": 3,
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

        metrics = estimate_metrics(X, y, pipe_args, estimator, args['num_cv'])

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
        json_fn = os.path.join(args['out_dir'], "{}-trial-{}.json".format(runname, trial.number))
        dump_json(json_fn, res)

        return np.mean(metrics['val_scores'])
       
    sampler = None
    if args['sampler'] == "tpe":
        sampler = optuna.samplers.TPESampler()
    elif args['sampler'] == "motpe":
        sampler = optuna.samplers.MOTPESampler()
    elif args['sampler'] == "random":
        sampler = optuna.samplers.RandomSampler()

    study = optuna.create_study(study_name=runname, sampler=sampler, direction="maximize")

    study.optimize(lambda trial: objective(trial, X, y, args, pipe_args, exp, runname),
                    n_trials=args['n_trials'], 
                    gc_after_trial=True)

    print(study.best_params)
    print(study.best_value)
    print(study.best_trial)
    print(study.trials_dataframe())
    if exp is not None:
        exp.log_table('optuna_trials.csv', study.trials_dataframe())
        exp.log_figure('Opt history', optuna.visualization.plot_optimization_history(study))
        exp.log_figure('Hyperparam importance', optuna.visualization.plot_param_importances(study))

    return
    
if __name__ == "__main__":
    main()
