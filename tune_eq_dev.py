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
import socket
import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
#from lightgbm import LGBMClassifier
import xgboost as xgb
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
import optuna
import time

from functools import partial

from SteveHelpers import dump_json, get_data_types
from SteveHelpers import SteveCorexWrapper
from SteveHelpers import SteveNumericNormalizer
from SteveHelpers import SteveAutoFeatLight
from SteveHelpers import SteveFeatureDropper
from SteveHelpers import SteveFeatureTyper
from SteveHelpers import SteveEncoder
from SteveHelpers import SteveNumericCapper
from SteveHelpers import SteveMissingIndicator
from SteveHelpers import SteveNumericImputer
from SteveHelpers import SteveCategoryImputer
from SteveHelpers import SteveCategoryCoalescer
from SteveHelpers import check_dataframe

def run_one_lgbm(X, y, pipe, args, num_boost_round, num_boost_round_final, params, exp=None, X_test=None):
    """
    pipe: preprocessing/FE pipeline to fit/use
    args: command line args
    num_boost_rounds: for CV
    num_boost_rounds: for final fit
    params: lgbm params
    exp: comet experiment object
    
    Returns:
    probas: predicted probabilies if train_full_model is true, else None
    preds: predicted classes if train_full_model is true, else None
    val_scores: estimated val score for each each CV fold
    train_scores: estimated val score for each each CV fold
    """
    probas = None
    preds = None
    val_scores = []
    train_scores = []
    best_iterations = []
    
    print("run_one_lgbm: args: {}".format(args))
    print("run_one_lgbm: params: {}".format(params))
    print("run_one_lgbm: num_boost_round (for cv): {}".format(num_boost_round))
    print("run_one_lgbm: num_boost_round_final: {}".format(num_boost_round_final))
          
    label_transformer = LabelEncoder()
    y = label_transformer.fit_transform(y)

    # Cross validation loop?
    if args.num_cv > 1:
        skf = StratifiedKFold(n_splits=args.num_cv, random_state=42, shuffle=True)

        cv_step = -1
        for train_index, val_index in skf.split(X, y):
            cv_step = cv_step + 1
            print("run_one_lgbm: cv_step {} of {}".format(cv_step, args.num_cv))
          
            X_train, X_val = X.loc[train_index], X.loc[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            pipe.fit(X_train, y_train)

            _X_train = pipe.transform(X_train)
            _X_val  = pipe.transform(X_val)
          
            check_dataframe(_X_train, "_X_train")
            check_dataframe(_X_val, "_X_val")
            
            dtrain = lgbm.Dataset(_X_train, label=y_train)
            dval   = lgbm.Dataset(_X_val, label=y_val)

            start = time.time()
            print("run_one_lgbm: cv lgbm train..")
            eval_r = {}
            _model = lgbm.train(params=params, 
                               train_set=dtrain, 
                               num_boost_round=num_boost_round,
                               valid_sets=[dval, dtrain],
                               valid_names=["val", "train"],
                               early_stopping_rounds=20,
                               evals_result=eval_r,
                               verbose_eval=50)
            train_time = (time.time() - start)
            print("..done lgbm training in {} secs".format(train_time))

            print("run_one_lgbm: calculating val metrics..")
            y_val_pred_proba = _model.predict(_X_val)
            y_val_pred = np.argmax(y_val_pred_proba,axis=1)
            val_micro_f1 = f1_score(y_val, y_val_pred, average="micro"),
            print(classification_report(y_val, y_val_pred, digits=4))

            print("run_one_lgbm: calculating train metrics..")
            y_train_pred_proba = _model.predict(_X_train)
            y_train_pred = np.argmax(y_train_pred_proba,axis=1)
            train_micro_f1 = f1_score(y_train, y_train_pred, average="micro"),
            print(classification_report(y_train, y_train_pred, digits=4))

            val_scores.append(val_micro_f1)
            train_scores.append(train_micro_f1)
            best_iterations.append(_model.best_iteration)
          

        def mean_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
            return m, m-h, m+h
              
        print("run_one_lgbm: cv complete.")
        print("run_one_lgbm: Train Scores: {}".format(train_scores))
        print("run_one_lgbm: Val Scores: {}".format(val_scores))
        print("run_one_lgbm: Val Score range: {}".format(mean_confidence_interval(val_scores)))
        print("run_one_lgbm: Best iterations: {}".format(best_iterations))
        print("run_one_lgbm: Best iteration range: {}".format(mean_confidence_interval(best_iterations)))

   
    if args.train_full_model == 1: 
        print("run_one_lgbm: Training full model.")
        pipe.fit(X, y)
        _X = pipe.transform(X)
        _X_test  = pipe.transform(X_test)

        dtrain = lgbm.Dataset(_X_train, label=y_train)

        if num_boost_round_final is None:
            if args.num_cv < 1:
                print("Error: num_boost_round_final is None, but CV was not run.")
            num_boost_round_final = np.rint(np.mean(best_iterations)*1.1).astype(int)
          
        print("run_one_lgbm: num_boost_round_final: {}".format(num_boost_round_final))
        _model = lgbm.train(params=params,  train_set=dtrain, 
                           num_boost_round=num_boost_round_final, verbose_eval=50)

        print("lgbm predict..")
        probas = _model.predict(_X_test)
        preds = np.argmax(probas, axis=1)
        preds = label_transformer.inverse_transform(preds)
          
    return probas, preds, val_scores, train_scores

def run_one_xgboost(X, y, pipe, args, num_boost_round, num_boost_round_final, params, exp=None, X_test=None):
    """
    pipe: preprocessing/FE pipeline to fit/use
    args: command line args
    num_boost_rounds: for CV
    num_boost_rounds: for final fit
    params: xgb params
    exp: comet experiment object
    
    Returns:
    probas: predicted probabilies if train_full_model is true, else None
    preds: predicted classes if train_full_model is true, else None
    val_scores: estimated val score for each each CV fold
    train_scores: estimated val score for each each CV fold
    """
    probas = None
    preds = None
    val_scores = []
    train_scores = []
    best_iterations = []
    
    print("run_one_xgboost: args: {}".format(args))
    print("run_one_xgboost: params: {}".format(params))
    print("run_one_xgboost: num_boost_round (for cv): {}".format(num_boost_round))
    print("run_one_xgboost: num_boost_round_final: {}".format(num_boost_round_final))
          
    label_transformer = LabelEncoder()
    y = label_transformer.fit_transform(y)

    # Cross validation loop?
    if args.num_cv > 1:
        skf = StratifiedKFold(n_splits=args.num_cv, random_state=42, shuffle=True)

        cv_step = -1
        for train_index, val_index in skf.split(X, y):
            cv_step = cv_step + 1
            print("run_one_xgboost: cv_step {} of {}".format(cv_step, args.num_cv))
          
            X_train, X_val = X.loc[train_index], X.loc[val_index]
            y_train, y_val = y[train_index], y[val_index]
            
            pipe.fit(X_train, y_train)

            _X_train = pipe.transform(X_train)
            _X_val  = pipe.transform(X_val)
          
            check_dataframe(_X_train, "_X_train")
            check_dataframe(_X_val, "_X_val")

            dtrain = xgb.DMatrix(_X_train, label=y_train)
            dval   = xgb.DMatrix(_X_val, label=y_val)

            start = time.time()
            print("run_one_xgboost: cv xgb train..")
            eval_r = {}
            _model = xgb.train(params=params, 
                               dtrain=dtrain, 
                               num_boost_round=num_boost_round,
                               evals=[(dtrain, "train"), (dval, "val")],
                               early_stopping_rounds=20,
                               evals_result=eval_r,
                               verbose_eval=50)
            train_time = (time.time() - start)
            print("..done xgb training in {} secs".format(train_time))

            print("run_one_xgboost: calculating val metrics..")
            y_val_pred_proba = _model.predict(dval, iteration_range=(0, _model.best_iteration))
            y_val_pred = np.argmax(y_val_pred_proba,axis=1)
            val_micro_f1 = f1_score(y_val, y_val_pred, average="micro"),
            print(classification_report(y_val, y_val_pred, digits=4))

            print("run_one_xgboost: calculating train metrics..")
            y_train_pred_proba = _model.predict(dtrain, iteration_range=(0, _model.best_iteration))
            y_train_pred = np.argmax(y_train_pred_proba,axis=1)
            train_micro_f1 = f1_score(y_train, y_train_pred, average="micro"),
            print(classification_report(y_train, y_train_pred, digits=4))

            val_scores.append(val_micro_f1)
            train_scores.append(train_micro_f1)
            best_iterations.append(_model.best_iteration)
          

        def mean_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
            return m, m-h, m+h
              
        print("run_one_xgboost: cv complete.")
        print("run_one_xgboost: Train Scores: {}".format(train_scores))
        print("run_one_xgboost: Val Scores: {}".format(val_scores))
        print("run_one_xgboost: Val Score range: {}".format(mean_confidence_interval(val_scores)))
        print("run_one_xgboost: Best iterations: {}".format(best_iterations))
        print("run_one_xgboost: Best iteration range: {}".format(mean_confidence_interval(best_iterations)))

   
    if args.train_full_model == 1: 
        print("run_one_xgboost: Training full model.")
        pipe.fit(X, y)
        _X = pipe.transform(X)
        _X_test  = pipe.transform(X_test)

        dtrain = xgb.DMatrix(_X, label=y)
        dtest  = xgb.DMatrix(_X_test)

        if num_boost_round_final is None:
            if args.num_cv < 1:
                print("Error: num_boost_round_final is None, but CV was not run.")
            num_boost_round_final = np.rint(np.mean(best_iterations)*1.1).astype(int)
          
        print("run_one_xgboost: num_boost_round_final: {}".format(num_boost_round_final))
        _model = xgb.train(params=params, 
                           dtrain=dtrain, 
                           num_boost_round=num_boost_round_final, 
                           xgb_model=None)

        print("xbg predict..")
        probas = _model.predict(dtest)
        preds = np.argmax(probas, axis=1)
        preds = label_transformer.inverse_transform(preds)
          
    return probas, preds, val_scores, train_scores

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Comet params
    parser.add_argument('--enable-comet', type=int, default=1)
    # Prep/FE params
    parser.add_argument('--geo-id-set', type=int, default=3)
    parser.add_argument('--autofeat', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--cat-encoder', type=int, default=2)
    
    # Search Params
    parser.add_argument('--run-type', type=str, default="optuna")
    
    # XGBoost/LGBM params
    parser.add_argument('--algo-set', type=int, default=1)
    parser.add_argument('--booster', type=str, default='gbtree')
    parser.add_argument('--grow-policy', type=str, default='lossguide')
    parser.add_argument('--scale-pos-weight', type=float, default=1.0)
    parser.add_argument('--num-cv', type=int, default=0)
    parser.add_argument('--train-full-model', type=int, default=0)
          
    
    # Optuna params
    parser.add_argument('--sampler', type=str, default='tpe')
    parser.add_argument('--n-trials', type=int, default=100)

    args = parser.parse_args()

    print("Command line arguments:")
    print(vars(args))

    id_col = 'building_id'
    target_col = 'damage_grade'
    out_dir = 'earthquake/out'
    data_id = '000'

    # Create an experiment with your api key
    exp=None
    if args.enable_comet == 1:
        exp = Experiment(
            project_name="eq_{}".format(args.run_type),
            workspace="stepthom",
            parse_args=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            log_graph=False
        )

    train_fn =  'earthquake/earthquake_train.csv'
    test_fn =  'earthquake/earthquake_test.csv'
    train_df  = pd.read_csv(train_fn)
    test_df  = pd.read_csv(test_fn)
    if args.sample_frac < 1.0:
        train_df  = train_df.sample(frac=args.sample_frac, random_state=3).reset_index(drop=True)
        
    X = train_df.drop([id_col, target_col], axis=1)
    y = train_df[target_col]
    X_test = test_df.drop([id_col], axis=1)

    geo_id_set = []
    if args.geo_id_set == 1:
        geo_id_set = ['geo_level_1_id']
    elif args.geo_id_set == 2:
        geo_id_set = ['geo_level_1_id', 'geo_level_2_id']
    elif args.geo_id_set == 3:
        geo_id_set = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

    prep1 = Pipeline([
        ('typer', SteveFeatureTyper(cols=geo_id_set, typestr='category'))
    ])

    prep1.fit(X)
    X = prep1.transform(X)
    X_test = prep1.transform(X_test)

    cat_cols, num_cols, bin_cols, float_cols, date_cols = get_data_types(X, id_col, target_col)
    
    steps = []

    if args.cat_encoder == 1:
        pass
    elif args.cat_encoder == 2:
        _cat_cols = list(set(cat_cols) - set(geo_id_set))
        steps.append(('cat_encoder', SteveEncoder(cols=_cat_cols,
                      encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32))))
        steps.append(('dropper', SteveFeatureDropper(_cat_cols)))
    elif args.cat_encoder == 3:
        _cat_cols = list(set(cat_cols) - set(geo_id_set))
        steps.append(('cat_encoder', SteveEncoder(cols=_cat_cols,
                       encoder=OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int32))))
        steps.append(('dropper', SteveFeatureDropper(_cat_cols)))

    steps.append(('num_capper', SteveNumericCapper(num_cols=['age'], max_val=30)))

    if args.autofeat == 1:
        steps.append(('num_autofeat', SteveAutoFeatLight(float_cols)))

    if args.normalize == 1:
        steps.append(('num_normalizer', SteveNumericNormalizer(float_cols, drop_orig=True)))
        
    enc =  ce.wrapper.PolynomialWrapper(
            ce.target_encoder.TargetEncoder(
                handle_unknown="value", 
                handle_missing="value", 
                min_samples_leaf=1, 
                smoothing=0.1, return_df=True))

    steps.append(("cat_enc2", enc))

    for step in steps:
        print(step)
    pipe = Pipeline(steps)

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
        exp.log_table('y_head.csv', y.head())
        exp.log_parameters(vars(args))
        exp.log_asset('SteveHelpers.py')

    if args.run_type == "one":
        probas = None
        preds = None
        val_scores = None
        train_scores = None
        
        
        if args.algo_set == 1:
            num_boost_round = 3500
            num_boost_round_final = 110
            lgbm_params = {
                "num_leaves": 2671,
                "min_child_weight": 0.7834887246590326,
                "min_data_in_leaf": 2,
                "learning_rate": 0.04670098249884267,
                "colsample_bylevel": 0.69404134745479,
                "colsample_bytree": 0.30337702784418485,
                "reg_alpha": 0.007004338652814285,
                "reg_lambda": 4.885400880425611,
                'extra_tree': False
            } 
                
            lgbm_params.update({
                  "bagging_freq": 0,
                  "bagging_fraction": 1.0,
                  "boosting": 'gbdt',
                  "max_depth": -1,
                  "n_jobs": 5,
                  "objective": "multiclass",
                  "verbosity": -1,
                  "num_class": 3,
                  "seed": 77,
                  "seed_per_iteration": True,
            })
            probas, preds, val_scores, train_scores = run_one_lgbm(
                X, y, pipe, args, num_boost_round, num_boost_round_final, lgbm_params, exp, X_test)
         
        if args.algo_set == 2:
            num_boost_round = 3500
            num_boost_round_final = 110
            xgb_params = {
                'max_leaves': 3949, 
                'min_child_weight': 1.1992742594922994, 
                'learning_rate': 0.003150897907094229, 
                'subsample': 0.47912852600183836, 
                'colsample_bylevel': 0.6961990874302973, 
                'colsample_bytree': 0.30862028608859277, 
                'reg_alpha': 1.8627000386165924, 
                'reg_lambda': 0.0186322334353944, 
                'gamma': 0.06465888061358417,
            }

            xgb_params.update({
                  "booster": 'gbtree',
                  "max_depth": 0,
                  "grow_policy": "lossguide",
                  "tree_method": "hist",
                  "n_jobs": 1,
                  "objective": "multi:softprob",
                  "eval_metric": "mlogloss",
                  "verbosity": 1,
                  "num_class": 3, 
                  "seed": 77,
                  "random_state": 77,
                  "seed_per_iteration": True,
            })

            probas, preds, val_scores, train_scores = run_one_xgboost(
                X, y, pipe, args, num_boost_round, num_boost_round_final, xgb_params, exp, X_test)
        
        print("val_scores: {}".format(val_scores))
        print("train_scores: {}".format(train_scores))
         
        if preds is not None:
            preds_df = pd.DataFrame(data={'id': test_df[id_col], target_col: preds})
            preds_fn = os.path.join(out_dir, "{}-{}-preds.csv".format(runname, data_id))
            preds_df.to_csv(preds_fn, index=False)
            print("tune_eq: Wrote preds file: {}".format(preds_fn))

        if probas is not None:
            probas_df = pd.DataFrame(probas, columns=["1", "2", "3"])
            probas_df[id_col] = test_df[id_col]
            probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
            probas_fn = os.path.join(out_dir, "{}-{}-probas.csv".format(runname, data_id))
            probas_df.to_csv(probas_fn, index=False)
            print("tune_eq: Wrote probas file: {}".format(probas_fn))
          
    if args.run_type == "optuna":
        
        def lgbm_objective(trial, X, y, pipe, exp):
           
            upper = 4096
            num_boost_round = 3500 
            num_leaves = trial.suggest_int("num_leaves", 4, upper, log=True)
            min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 2, 2**7, log=True)
            min_child_weight = trial.suggest_loguniform("min_child_weight", 0.001, 128)
            learning_rate = trial.suggest_loguniform("learning_rate", 1/1024, 1.0)
            subsample = trial.suggest_float("subsample", 0.1, 1.0)
            feature_fraction_bynode = trial.suggest_float("feature_fraction_bynode", 0.01, 1.0)
            feature_fraction = trial.suggest_float("feature_fraction", 0.01, 1.0)
            reg_alpha = trial.suggest_loguniform("reg_alpha", 1/1024, 1024)
            reg_lambda = trial.suggest_loguniform("reg_lambda", 1/1024, 1024)
            extra_trees = trial.suggest_categorical("extra_trees", [True, False])
            
            lgbm_params = {
                  "num_leaves": num_leaves,
                  "min_child_weight": min_child_weight,
                  "min_data_in_leaf": min_data_in_leaf,
                  "learning_rate": learning_rate,
                  "feature_fraction_bynode": feature_fraction_bynode,
                  "feature_fraction": feature_fraction,
                  "reg_alpha": reg_alpha,
                  "reg_lambda": reg_lambda,
                  "extra_trees": extra_trees,
                
                  "bagging_freq": 0,
                  "bagging_fraction": 1.0,
                  "boosting": 'gbdt',
                  "max_depth": -1,
                  "n_jobs": 5,
                  "objective": "multiclass",
                  "verbosity": -1,
                  "num_class": 3,
                  "seed": 77,
                  "seed_per_iteration": True,
            }

            _, _, val_scores, train_scores = run_one_lgbm(
                X, y, pipe, args, num_boost_round, num_boost_round_final=None, params=lgbm_params, exp=exp, X_test=None)
          
            if exp is not None:
                exp.log_metric("mean_val_score", np.mean(val_scores), step=trial.number)
                exp.log_metric("mean_train_score", np.mean(train_scores), step=trial.number)
                exp.log_text(lgbm_params, step=trial.number)
            
            return np.mean(val_scores)
       
        def xgb_objective(trial, X, y, pipe, exp):
            upper = 4096
            num_boost_round = 2500 # trial.suggest_int("num_boost_round", 4, upper, log=True)
            max_leaves = trial.suggest_int("max_leaves", 4, upper, log=True)
            min_child_weight = trial.suggest_loguniform("min_child_weight", 0.001, 128)
            learning_rate = trial.suggest_loguniform("learning_rate", 1/1024, 1.0)
            subsample = trial.suggest_float("subsample", 0.1, 1.0)
            colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.01, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.01, 1.0)
            reg_alpha = trial.suggest_loguniform("reg_alpha", 1/1024, 1024)
            reg_lambda = trial.suggest_loguniform("reg_lambda", 1/1024, 1024)
            gamma = trial.suggest_loguniform("gamma", 1/1024, 128)
            
            xgb_params = {
                  "max_leaves": max_leaves,
                  "min_child_weight": min_child_weight,
                  "learning_rate": learning_rate,
                  "subsample": subsample,
                  "colsample_bylevel": colsample_bylevel,
                  "colsample_bytree": colsample_bytree,
                  "reg_alpha": reg_alpha,
                  "reg_lambda": reg_lambda,
                  "gamma": gamma,
                
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
          
            _, _, val_scores, train_scores = run_one_xgboost(
                X, y, pipe, args, num_boost_round, num_boost_round_final=None, params=xgb_params, exp=exp, X_test=None)
          
            if exp is not None:
                exp.log_metric("mean_val_score", np.mean(val_scores), step=trial.number)
                exp.log_metric("mean_train_score", np.mean(train_scores), step=trial.number)
                exp.log_text(xgb_params, step=trial.number)
            
            return np.mean(val_scores)
       
        sampler = None
        if args.sampler == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif args.sampler == "motpe":
            sampler = optuna.samplers.MOTPESampler()
        elif args.sampler == "random":
            sampler = optuna.samplers.RandomSampler()
        
        study = optuna.create_study(study_name=runname, sampler=sampler, direction="maximize")
        
        if args.algo_set == 1:
            study.optimize(lambda trial: lgbm_objective(trial, X, y, pipe, exp),
                           n_trials=args.n_trials, 
                           gc_after_trial=True)
        elif args.algo_set == 2:
            study.optimize(lambda trial: xgb_objective(trial, X, y, pipe, exp), 
                           n_trials=args.n_trials, 
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
