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

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Comet params
    parser.add_argument('--enable-comet', type=int, default=1)
    # Prep/FE params
    parser.add_argument('--geo-id-set', type=int, default=3)
    #parser.add_argument('--min-sample-leaf', type=int, default=5)
    #parser.add_argument('--smoothing', type=float, default=10.0)
    parser.add_argument('--autofeat', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--sample-frac', type=float, default=1.0)
    #parser.add_argument('--ensemble', type=int, default=0)
    parser.add_argument('--cat-encoder', type=int, default=1)
    
    # Search Params
    parser.add_argument('--run-type', type=str, default="optuna")
    #parser.add_argument('--time-budget', type=int, default=60)
    parser.add_argument('--metric', type=str, default="micro_f1")
    
    # XGBoost/LGBM params
    parser.add_argument('--algo-set', type=int, default=1)
    parser.add_argument('--booster', type=str, default='gbtree')
    parser.add_argument('--grow-policy', type=str, default='lossguide')
    parser.add_argument('--scale-pos-weight', type=float, default=1.0)
    parser.add_argument('--num-cv', type=int, default=12)
    
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
            project_name="eq_searcher",
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


    geo_id_set = []
    if args.geo_id_set == 1:
        geo_id_set = ['geo_level_1_id']
    elif args.geo_id_set == 2:
        geo_id_set = ['geo_level_1_id', 'geo_level_2_id']
    else:
        geo_id_set = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

    estimator_list = ['lgbm']
    if args.algo_set == 1:
        estimator_list = ['lgbm']
    elif args.algo_set == 2:
        estimator_list = ['xgboost']
    elif args.algo_set == 3:
        estimator_list = ['catboost']
    elif args.algo_set == 4:
        estimator_list = ['rf']
    elif args.algo_set == 5:
        estimator_list = ['extra_tree']

    X = train_df.drop([id_col, target_col], axis=1)
    y = train_df[target_col]
    X_test = test_df.drop([id_col], axis=1)

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
            ce.target_encoder.TargetEncoder(handle_unknown="value", handle_missing="value", min_samples_leaf=1, smoothing=0.1, return_df=True))

    steps.append(("cat_enc2", enc))

    print(steps)
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

    feat_imp = None
    if args.run_type == "flaml":
        from flaml import AutoML
        os.environ['OS_STEVE_MIN_SAMPLE_LEAF'] = str(args.min_sample_leaf)
        os.environ['OS_STEVE_SMOOTHING'] = str(args.smoothing)
        os.environ['OS_STEVE_BOOSTING'] = str(args.booster)
       
        automl_settings = {
            "time_budget": args.time_budget,
            "task": 'classification',
            "n_jobs": 5,
            "estimator_list": estimator_list,
            "eval_method": "cv",
            "n_splits": 3,
            "metric": args.metric,
            "ensemble": False,
        }
        automl_config = {
            "comet_exp": exp,
            "verbose": 1,
            "log_training_metric": True,
            "model_history": False,
            "log_file_name": "logs/flaml-{}.log".format(runname),
        }

        print("automl_settings:")
        print(automl_settings)
        print("automl_config:")
        print(automl_config)

        # Log some things before fit(), to more-easily monitor runs
        if exp is not None:
            exp.log_parameters(automl_settings)
            exp.log_parameter("metric_name", args.metric)

        clf = AutoML()
        clf.fit(X, y, **automl_config, **automl_settings)
        best_loss = clf.best_loss

        print("Best config")
        bm = clf.best_model_for_estimator(clf.best_config)
        print("Best model")
        bm = clf.best_model_for_estimator(clf.best_estimator)
        print(bm.model)
        feature_names = bm.feature_names_
        feat_imp =  pd.DataFrame({'Feature': feature_names, 'Importance': bm.model.feature_importances_}).sort_values('Importance', ascending=False)
        print("Feature importances")
        print(feat_imp.head())

    elif args.run_type == "optuna":
        
        def lgbm_objective(trial, X, y, booster="gbtree", scale_pos_weight=1):
           
            upper = 4096
            num_boost_round = 3500 # trial.suggest_int("num_boost_round", 4, upper, log=True)
            num_leaves = trial.suggest_int("num_leaves", 4, upper, log=True)
            min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 2, 2**7, log=True)
            min_child_weight = trial.suggest_loguniform("min_child_weight", 0.001, 128)
            learning_rate = trial.suggest_loguniform("learning_rate", 1/1024, 1.0)
            subsample = trial.suggest_float("subsample", 0.1, 1.0)
            colsample_bylevel = trial.suggest_float("colsample_by_level", 0.01, 1.0)
            colsample_bytree = trial.suggest_float("colsample_by_tree", 0.01, 1.0)
            reg_alpha = trial.suggest_loguniform("reg_alpha", 1/1024, 1024)
            reg_lambda = trial.suggest_loguniform("reg_lambda", 1/1024, 1024)
            #gamma = trial.suggest_loguniform("gamma", 1/1024, 128)
            extra_tree = trial.suggest_categorical("extra_tree", [True, False])
            
            bagging_freq = 0
            bagging_fraction = 1.0
            #if booster=="rf":
                #bagging_freq = 5
                
            params = {
                  "num_leaves": num_leaves,
                  "min_child_weight": min_child_weight,
                  "min_data_in_leaf": min_data_in_leaf,
                  "learning_rate": learning_rate,
                  "colsample_bynode": colsample_bylevel,
                  "colsample_bytree": colsample_bytree,
                  "reg_alpha": reg_alpha,
                  "reg_lambda": reg_lambda,
                  "bagging_freq": bagging_freq,
                  "bagging_fraction": bagging_fraction,
                  "extra_trees": extra_tree,
                  "boosting": booster,
                  "max_depth": -1,
                  "n_jobs": 5,
                  "objective": "multiclass",
                  "verbosity": -1,
                  "num_class": 3,
                  "seed": 77,
            }

            label_transformer = LabelEncoder()
            y = label_transformer.fit_transform(y)
           
            # Cross validation loop
            skf = StratifiedKFold(n_splits=args.num_cv, random_state=42, shuffle=True)

            val_scores = []
            train_scores = []
            best_iterations = []
            for train_index, val_index in skf.split(X, y):
                X_train, X_val = X.loc[train_index], X.loc[val_index]
                y_train, y_val = y[train_index], y[val_index]

                pipe.fit(X_train, y_train)

                _X_train = pipe.transform(X_train)
                _X_val  = pipe.transform(X_val)

                dtrain = lgbm.Dataset(_X_train, label=y_train)
                dval   = lgbm.Dataset(_X_val, label=y_val)

                start = time.time()
                print("lgbm train..")
                eval_r = {}
                _model = lgbm.train(params=params, train_set=dtrain, num_boost_round=num_boost_round, 
                                   valid_sets=[dval, dtrain],
                                   valid_names=["val", "train"],
                                   early_stopping_rounds=20,
                                   evals_result=eval_r,
                                   verbose_eval=50,
                                   init_model=None)
                train_time = (time.time() - start)
                print("..done. {} secs".format(train_time))


                y_pred_proba = _model.predict(_X_val)
                y_pred = np.argmax(y_pred_proba,axis=1)
                print(classification_report(y_val, y_pred, digits=4))

                custom_metrics = {
                    'val_log_loss':  log_loss(y_val, y_pred_proba),
                    'val_micro_f1':  f1_score(y_val, y_pred, average="micro"),
                }

                y_pred_proba = _model.predict(_X_train)
                y_pred = np.argmax(y_pred_proba,axis=1)

                custom_metrics.update({
                    'train_log_loss':  log_loss(y_train, y_pred_proba),
                    'train_micro_f1':  f1_score(y_train, y_pred, average="micro"),
                })
                print(classification_report(y_train, y_pred, digits=4))

                custom_metrics.update({
                    "train_seconds": train_time,
                    "step": trial.number,
                })

                print("Fold metrics:")
                print(custom_metrics)

                val_scores.append(custom_metrics['val_micro_f1'])
                train_scores.append(custom_metrics['train_micro_f1'])
                best_iterations.append(_model.best_iteration)
                
            print("Train Scores: {}".format(train_scores))
            print("Val Scores: {}".format(val_scores))
            print("Best iterations: {}".format(best_iterations))
                
            if exp is not None:
                exp.log_metric("mean_val_score", np.mean(val_scores), step=trial.number)
                exp.log_metric("mean_train_score", np.mean(train_scores), step=trial.number)
                exp.log_text(params, step=trial.number)
                exp.log_text(num_boost_round, step=trial.number)
                
            # Now train final model on full dataset; output preds
            pipe.fit(X, y)
            _X_train = pipe.transform(X)
            y_train = y
            _X_test  = pipe.transform(X_test)

            dtrain = lgbm.Dataset(_X_train, label=y_train)
            start = time.time()
            print("lgbm train..")
            eval_r = {}
            num_boost_round_final = num_boost_round
            if len(best_iterations) > 1:
                num_boost_round_final = np.rint(np.mean(best_iterations)).astype(int)
            num_boost_round_final = num_boost_round_final+300
            print("num_boost_round_final: {}".format(num_boost_round_final))
            _model = lgbm.train(params=params, train_set=dtrain, num_boost_round=num_boost_round_final,
                               verbose_eval=50,
                               init_model=None)
            train_time = (time.time() - start)
            print("..done. {} secs".format(train_time))

            print("lgbm predict..")
            probas = _model.predict(_X_test)
            preds = np.argmax(probas, axis=1)
            preds = label_transformer.inverse_transform(preds)

            print("outputting files..")
            preds_df = pd.DataFrame(data={'id': test_df[id_col], target_col: preds})
            preds_fn = os.path.join(out_dir, "{}-{}-{}-preds.csv".format(runname, data_id, trial.number))
            preds_df.to_csv(preds_fn, index=False)
            print("tune_eq: Wrote preds file: {}".format(preds_fn))

            probas_df = pd.DataFrame(probas, columns=["1", "2", "3"])
            probas_df[id_col] = test_df[id_col]
            probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
            probas_fn = os.path.join(out_dir, "{}-{}-{}-probas.csv".format(runname, data_id, trial.number))
            probas_df.to_csv(probas_fn, index=False)
            print("tune_eq: Wrote probas file: {}".format(probas_fn))
            
            return np.mean(val_scores)
       
        def xgb_objective(trial, X, y, booster="gbtree", grow_policy="lossguide", scale_pos_weight=1):
           
            upper = 4096
            num_boost_round = 2500 # trial.suggest_int("num_boost_round", 4, upper, log=True)
            max_leaves = trial.suggest_int("max_leaves", 4, upper, log=True)
            min_child_weight = trial.suggest_loguniform("min_child_weight", 0.001, 128)
            learning_rate = trial.suggest_loguniform("learning_rate", 1/1024, 1.0)
            subsample = trial.suggest_float("subsample", 0.1, 1.0)
            colsample_bylevel = trial.suggest_float("colsample_by_level", 0.01, 1.0)
            colsample_bytree = trial.suggest_float("colsample_by_tree", 0.01, 1.0)
            reg_alpha = trial.suggest_loguniform("reg_alpha", 1/1024, 1024)
            reg_lambda = trial.suggest_loguniform("reg_lambda", 1/1024, 1024)
            gamma = trial.suggest_loguniform("gamma", 1/1024, 128)
            #booster = trial.suggest_categorical("booster", ["gbtree", "dart"])
           
            max_depth = 0
            if grow_policy == "depthwise":
                max_depth = trial.suggest_int("max_depth", 4, 32)
            
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
                  "booster": booster,
                  "max_depth": max_depth,
                  "grow_policy": grow_policy,
                  #"scale_pos_weight": scale_pos_weight,
                  "tree_method": "hist",
                  "n_jobs": 5,
                  "objective": "multi:softprob",
                  "eval_metric": "mlogloss",
                  "verbosity": 1,
                  "num_class": 3,
            }

            label_transformer = LabelEncoder()
            y = label_transformer.fit_transform(y)
           
            # Cross validation loop
            skf = StratifiedKFold(n_splits=args.num_cv, random_state=42, shuffle=True)

            val_scores = []
            train_scores = []
            best_iterations = []
            for train_index, val_index in skf.split(X, y):
                X_train, X_val = X.loc[train_index], X.loc[val_index]
                y_train, y_val = y[train_index], y[val_index]

                pipe.fit(X_train, y_train)

                _X_train = pipe.transform(X_train)
                _X_val  = pipe.transform(X_val)

                dtrain = xgb.DMatrix(_X_train, label=y_train)
                dval   = xgb.DMatrix(_X_val, label=y_val)

                start = time.time()
                print("xbg train..")
                eval_r = {}
                _model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=num_boost_round,
                                   evals=[(dtrain, "train"), (dval, "val")],
                                   early_stopping_rounds=20,
                                   evals_result=eval_r,
                                   verbose_eval=50,
                                   xgb_model=None)
                train_time = (time.time() - start)
                print("..done. {} secs".format(train_time))


                y_pred_proba = _model.predict(dval, iteration_range=(0, _model.best_iteration))
                y_pred = np.argmax(y_pred_proba,axis=1)
                print(classification_report(y_val, y_pred, digits=4))

                custom_metrics = {
                    'val_log_loss':  log_loss(y_val, y_pred_proba),
                    'val_micro_f1':  f1_score(y_val, y_pred, average="micro"),
                    #'val_macro_f1':  f1_score(y_val, y_pred, average="macro"),
                    #'val_weighted_f1':  f1_score(y_val, y_pred, average="weighted"),
                    #'val_roc_auc':  roc_auc_score(y_val, y_pred_proba, multi_class="ovo"),
                }

                y_pred_proba = _model.predict(dtrain, iteration_range=(0, _model.best_iteration))
                y_pred = np.argmax(y_pred_proba,axis=1)

                custom_metrics.update({
                    'train_log_loss':  log_loss(y_train, y_pred_proba),
                    'train_micro_f1':  f1_score(y_train, y_pred, average="micro"),
                    #'train_macro_f1':  f1_score(y_train, y_pred, average="macro"),
                    #'train_weighted_f1':  f1_score(y_train, y_pred, average="weighted"),
                    #'train_roc_auc':  roc_auc_score(y_train, y_pred_proba, multi_class="ovo"),
                })
                print(classification_report(y_train, y_pred, digits=4))

                custom_metrics.update({
                    "train_seconds": train_time,
                    "step": trial.number,
                })

                print("Fold metrics:")
                print(custom_metrics)

                val_scores.append(custom_metrics['val_micro_f1'])
                train_scores.append(custom_metrics['train_micro_f1'])
                best_iterations.append(_model.best_iteration)
                
                
            print("Train Scores: {}".format(train_scores))
            print("Val Scores: {}".format(val_scores))
            print("Best iterations: {}".format(best_iterations))
                
            if exp is not None:
                exp.log_metric("mean_val_score", np.mean(val_scores), step=trial.number)
                exp.log_metric("mean_train_score", np.mean(train_scores), step=trial.number)
                exp.log_text(xgb_params, step=trial.number)
                exp.log_text(num_boost_round, step=trial.number)
                
            
            # Now train final model on full dataset; output preds
            pipe.fit(X, y)
            _X_train = pipe.transform(X)
            y_train = y
            _X_test  = pipe.transform(X_test)

            dtrain = xgb.DMatrix(_X_train, label=y_train)
            dtest  = xgb.DMatrix(_X_test)

            start = time.time()
            print("xbg train, one last time..")
            num_boost_round_final = num_boost_round
            if len(best_iterations) > 1:
                num_boost_round_final = np.rint(np.mean(best_iterations)).astype(int)
            num_boost_round_final = num_boost_round_final+300
            print("num_boost_round_final: {}".format(num_boost_round_final))
            _model = xgb.train(params=xgb_params, 
                               dtrain=dtrain, 
                               num_boost_round=num_boost_round_final, 
                               xgb_model=None,
                               #evals=[(dtrain, "train"), (dval, "val")],
                               #early_stopping_rounds=20,
                               verbose_eval=50,
                            )

            print("xbg predict..")
            probas = _model.predict(dtest,  iteration_range=(0,_model.best_iteration))
            preds = np.argmax(probas, axis=1)
            preds = label_transformer.inverse_transform(preds)

            print("outputting files..")
            preds_df = pd.DataFrame(data={'id': test_df[id_col], target_col: preds})
            preds_fn = os.path.join(out_dir, "{}-{}-{}-preds.csv".format(runname, data_id, trial.number))
            preds_df.to_csv(preds_fn, index=False)
            print("tune_eq: Wrote preds file: {}".format(preds_fn))

            probas_df = pd.DataFrame(probas, columns=["1", "2", "3"])
            probas_df[id_col] = test_df[id_col]
            probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
            probas_fn = os.path.join(out_dir, "{}-{}-{}-probas.csv".format(runname, data_id, trial.number))
            probas_df.to_csv(probas_fn, index=False)
            print("tune_eq: Wrote probas file: {}".format(probas_fn))
            
            
            return np.mean(val_scores)
       
        sampler = None
        if args.sampler == "tpe":
            sampler = optuna.samplers.TPESampler()
        elif args.sampler == "motpe":
            sampler = optuna.samplers.MOTPESampler()
        elif args.sampler == "random":
            sampler = optuna.samplers.RandomSampler()
        else:
            pass
        
        study = optuna.create_study(study_name=runname, sampler=sampler, direction="maximize")
        
        if args.algo_set == 1:
            study.optimize(lambda trial: lgbm_objective(trial, X, y, 
                                                   booster="gbdt",
                                                   scale_pos_weight=args.scale_pos_weight,), 
                           n_trials=args.n_trials, 
                           gc_after_trial=True)
        elif args.algo_set == 2:
            study.optimize(lambda trial: xgb_objective(trial, X, y, 
                                                   booster="gbtree",
                                                   grow_policy="lossguide",
                                                   scale_pos_weight=args.scale_pos_weight,), 
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
    
    else:
        pass

    endtime = datetime.datetime.now()
    duration = (endtime - starttime).seconds

if __name__ == "__main__":
    main()
