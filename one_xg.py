import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# import comet_ml at the top of your file
from comet_ml import Experiment
import uuid
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot
import os
import sys

import json
import socket
import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

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
import xgboost as xgb
from sklearn.model_selection import train_test_split
import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
import time

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Comet params
    parser.add_argument('--enable-comet', type=int, default=1)
    
    # Prep/FE params
    parser.add_argument('--geo-id-set', type=int, default=3)
    parser.add_argument('--n-hidden', type=int, default=4)
    parser.add_argument('--dim-hidden', type=int, default=2)
    parser.add_argument('--smooth-marginals', type=int, default=0)
    parser.add_argument('--min-sample-leaf', type=int, default=5)
    parser.add_argument('--smoothing', type=float, default=10.0)
    parser.add_argument('--autofeat', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--ensemble', type=int, default=0)
    parser.add_argument('--cat-encoder', type=int, default=2)
    
    parser.add_argument('--golden-params-id', type=int, default=1)
    parser.add_argument('--booster', type=str, default='gbtree')
    parser.add_argument('--grow-policy', type=str, default='lossguide')
    
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
            project_name="eq_one",
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
    run_fn = os.path.join(out_dir, "tune_eq_{}.json".format(runname))
    print("tune_eq: Run name: {}".format(runname))
    

    starttime = datetime.datetime.now()
    feat_imp = None
    
  
    
    # One 
    num_boost_round = 3906
    xgb_params = {
          "max_leaves": 663,
          "min_child_weight": 0.9037392121603112,
          "learning_rate": 0.004360214874813832,
          "subsample": 0.6646042224958224,
          "colsample_bylevel": 1,
          "colsample_bytree": 0.22134635270497466,
          "reg_alpha": 0.005543980767407033,
          "reg_lambda": 0.0031402829138070416,
          "gamma": 0,
    }
    
    if args.golden_params_id == 2:
        num_boost_round = 1310
        xgb_params =  {
            'max_leaves': 3481, 
            'min_child_weight': 0.009094914108923881, 
            'learning_rate': 0.00942470915870628, 
            'subsample': 0.9613994010078177, 
            'colsample_by_level': 0.917443995776857, 
            'colsample_by_tree': 0.2398890012561789, 
            'reg_alpha': 0.47215464811770447, 
            'reg_lambda': 0.1224498181695237, 
            'gamma': 0.0028804673824537946,
        }
    elif args.golden_params_id == 3:
        num_boost_round = 2355
        xgb_params =  {
            'max_leaves': 2135, 
            'min_child_weight': 0.014188330852854937, 
            'learning_rate': 0.004377366972868391, 
            'subsample': 0.9362761453834317, 
            'colsample_by_level': 0.9383760173638238, 
            'colsample_by_tree': 0.23846484953336966, 
            'reg_alpha': 0.7932429728688718, 
            'reg_lambda': 0.0905671609396819, 
            'gamma': 0.003548273713177225,
        }

   
    xgb_params.update({
          "booster": args.booster,
          "max_depth": 0,
          "grow_policy": "lossguide",
          "tree_method": "hist",
          "n_jobs": 3,
          "objective": "multi:softprob",
          "eval_metric": "mlogloss",
          "verbosity": 1,
          "num_class": 3, 
    })
    
    print("xgb_params: {}".format(xgb_params))

    label_transformer = LabelEncoder()
    y = label_transformer.fit_transform(y)
    
    if exp is not None:
        exp.log_other('train_fn', train_fn)
        exp.log_other('test_fn', test_fn)
        exp.log_table('X_head.csv', X.head())
        exp.log_parameters(vars(args))
        exp.log_parameters(xgb_params)
        exp.log_parameter("num_boost_round", num_boost_round)
        exp.log_asset('SteveHelpers.py')
    
    val_scores = []
    train_scores = []
    best_iterations = []
    cv_step = 0
    if True:
        # Cross validation loop
        skf = StratifiedKFold(n_splits=8, random_state=77, shuffle=True)

        for train_index, val_index in skf.split(X, y):
            cv_step = cv_step+1

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
                               xgb_model=None)
            train_time = (time.time() - start)
            print("..done. {} secs".format(train_time))

            y_pred_proba = _model.predict(dval, iteration_range=(0, _model.best_iteration))
            y_pred = np.argmax(y_pred_proba,axis=1)
            print("Val:")
            print(classification_report(y_val, y_pred, digits=4))

            custom_metrics = {
                'val_log_loss':  log_loss(y_val, y_pred_proba),
                'val_micro_f1':  f1_score(y_val, y_pred, average="micro"),
                'val_macro_f1':  f1_score(y_val, y_pred, average="macro"),
                'val_weighted_f1':  f1_score(y_val, y_pred, average="weighted"),
                'val_roc_auc':  roc_auc_score(y_val, y_pred_proba, multi_class="ovo"),
            }
            
            if exp is not None:
                exp.log_confusion_matrix(y_val, y_pred, title="Val Confusion Matrix {}".format(cv_step))

            y_pred_proba = _model.predict(dtrain, iteration_range=(0, _model.best_iteration))
            y_pred = np.argmax(y_pred_proba,axis=1)

            custom_metrics.update({
                'train_log_loss':  log_loss(y_train, y_pred_proba),
                'train_micro_f1':  f1_score(y_train, y_pred, average="micro"),
                'train_macro_f1':  f1_score(y_train, y_pred, average="macro"),
                'train_weighted_f1':  f1_score(y_train, y_pred, average="weighted"),
                'train_roc_auc':  roc_auc_score(y_train, y_pred_proba, multi_class="ovo"),
            })
            print("Train:")
            print(classification_report(y_train, y_pred, digits=4))

            custom_metrics.update({
                "train_seconds": train_time,
                "best_iteration": _model.best_iteration,
                "cv_step": cv_step,
            })

            print("Fold metrics:")
            print(custom_metrics)
            print("Best ntree_limit:")
            print(_model.best_ntree_limit)
            print("Best score:")
            print(_model.best_score)
            print("Best iteration:")
            print(_model.best_iteration)

            val_scores.append(custom_metrics['val_micro_f1'])
            train_scores.append(custom_metrics['train_micro_f1'])
            best_iterations.append(_model.best_iteration)
            
            feature_important = _model.get_score(importance_type='weight')
            keys = list(feature_important.keys())
            values = list(feature_important.values())
            feat_imp = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
            print("Feature Importances:")
            print(feat_imp.head(20))
            
            # Learning Curves
            curve = pd.DataFrame({
                'Train Loss': eval_r['train']['mlogloss'],
                'Val Loss': eval_r['val']['mlogloss'],
                })
            
            if exp is not None:
                exp.log_metrics(custom_metrics, step=cv_step)
                exp.log_table('feat_imp_{}.csv'.format(cv_step), feat_imp)
                exp.log_table('learning_curve_{}.csv'.format(cv_step), curve)
                exp.log_metric("mean_val_score", np.mean(val_scores))
                exp.log_metric("mean_train_score", np.mean(train_scores))
                exp.log_metric("mean_best_iteration", np.mean(best_iterations))

        print("Train Scores: {} {}".format(np.mean(train_scores), train_scores))
        print("Val Scores: {} {}".format(np.mean(val_scores), val_scores))
        print("Best Iteration: {} {}".format(np.mean(best_iterations), best_iterations))
        if exp is not None:
            exp.log_metric("mean_val_score", np.mean(val_scores))
            exp.log_metric("mean_train_score", np.mean(train_scores))
            exp.log_metric("mean_best_iteration", np.mean(best_iterations))

    pipe.fit(X, y)
    _X = pipe.transform(X)
    _X_test = pipe.transform(X_test)

    dtrain = xgb.DMatrix(_X, label=y)
    dtest  = xgb.DMatrix(_X_test)

    start = time.time()
    print("xbg train, one last time..")
    num_boost_round_final = num_boost_round
    if len(best_iterations) > 1:
        num_boost_round_final = np.rint(np.mean(best_iterations))
    print("num_boost_round_final: {}".format(num_boost_round_final))
    _model = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=num_boost_round_final, xgb_model=None)
    
    probas = _model.predict(dtest, ntree_limit=num_boost_round_final)
    preds = np.argmax(probas, axis=1)
    preds = label_transformer.inverse_transform(preds)
    
    preds_df = pd.DataFrame(data={'id': test_df[id_col], target_col: preds})
    preds_fn = os.path.join(out_dir, "{}-{}-preds.csv".format(runname, data_id))
    preds_df.to_csv(preds_fn, index=False)
    print("tune_eq: Wrote preds file: {}".format(preds_fn))

    probas_df = pd.DataFrame(probas, columns=["1", "2", "3"])
    probas_df[id_col] = test_df[id_col]
    probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
    probas_fn = os.path.join(out_dir, "{}-{}-probas.csv".format(runname, data_id))
    probas_df.to_csv(probas_fn, index=False)
    print("tune_eq: Wrote probas file: {}".format(probas_fn))

if __name__ == "__main__":
    main()
