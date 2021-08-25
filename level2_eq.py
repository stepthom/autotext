import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
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

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
#from mlxtend import StackingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from functools import partial

from SteveHelpers import dump_json, get_data_types
from EQHelpers import get_pipeline_steps, run_one

from sklearn.compose import make_column_selector
          
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--id-col', type=str, default="building_id")
    parser.add_argument('--target-col', type=str, default="damage_grade")
    parser.add_argument('--out-dir', type=str, default="earthquake/out")
    
    # Prep/FE params
    parser.add_argument('--sample-frac', type=float, default=1.0)
    
    args = parser.parse_args()
    args = vars(args)

    print("Command line arguments:")
    print(args)

    data_id = '000'

    train_fn = 'earthquake/earthquake_train.csv'
    test_fn = 'earthquake/earthquake_test.csv'
    train_df = pd.read_csv(train_fn)
    test_df = pd.read_csv(test_fn)
    if args['sample_frac']< 1.0:
        train_df  = train_df.sample(frac=args['sample_frac'], random_state=3).reset_index(drop=True)
        
    X = train_df.drop([args['id_col'], args['target_col']], axis=1)
    X_test = test_df.drop([args['id_col']], axis=1)
    y = train_df[args['target_col']]
    label_transformer = LabelEncoder()
    y = label_transformer.fit_transform(y)
    
    trial_logs = [
        # lgbm, 0.7553, 0.7533
        "earthquake/out/583b7931b2274e45839372114f3dc80f-trial-43.json",
        
        # lgbm, 0.7544, 0.7519
        #"earthquake/out/583b7931b2274e45839372114f3dc80f-trial-21.json",
        
        # lgbm, 0.7548, 
        #"earthquake/out/583b7931b2274e45839372114f3dc80f-trial-22.json",
        
        # xgboost, 0.7516, 0.7437
        #"earthquake/out/b7bce481fcab40de88418a34fc05f4c0-trial-26.json",
        
        # hist, 0.7480, 0.7458
        #"earthquake/out/0a7930f91e194ed2b0d18b9a01352b31-trial-67.json", 
    ]
    
    estimators = []
    
    i = 0
    for trial_log in trial_logs:
        i = i + 1
        run = {}
        with open(trial_log) as f:
            try:
                run = json.load(f)
            except JSONDecodeError as e:
                print("ERROR: cannot parse json file {}".format(run_file))
                print(e)
       
        args = run['args']
        pipe_args = run['pipe_args']
        params = run['params']
        metrics = run['metrics']
        runname = run['runname']
        trial = run['trial']
        
        n_estimators = None
        bi = metrics.get('best_iterations_range', [])
        if len(bi) > 0:
            
            # This seems to be about best
            n_estimators = np.floor(max(bi)).astype(int)
            
        estimator = None
        if args['estimator_name'] == "lgbm":
            params['n_estimators'] = n_estimators
            estimator = LGBMClassifier(**params)
        elif args['estimator_name'] == "xgboost":
            params['n_estimators'] = n_estimators
            estimator = XGBClassifier(**params)
        elif args['estimator_name'] == "rf":
            estimator = RandomForestClassifier(**params)
        elif args['estimator_name'] == "lr":
            estimator = LogisticRegression(**params)
        elif args['estimator_name'] == "hist":
            estimator = HistGradientBoostingClassifier(**params)
        else:
            print("Unknown estimator name {}".format(estimator_name))
                           
        pipe, estimator = run_one(X, y, pipe_args, estimator)
                                             
        _X_test = pipe.transform(X_test)
        probas = estimator.predict_proba(_X_test)
        preds  = estimator.predict(_X_test)
        preds = label_transformer.inverse_transform(preds)

        preds_df = pd.DataFrame(data={'id': test_df[args['id_col']], args['target_col']: preds})
        preds_fn = os.path.join(args['out_dir'], "{}-{}-preds.csv".format(runname, trial))
        preds_df.to_csv(preds_fn, index=False)
        print("level2: Wrote preds file: {}".format(preds_fn))

        probas_df = pd.DataFrame(probas, columns=["1", "2", "3"])
        probas_df[args['id_col']] = test_df[args['id_col']]
        probas_df = probas_df[ [args['id_col']] + [ col for col in probas_df.columns if col != args['id_col'] ] ]
        probas_fn = os.path.join(args['out_dir'], "{}-{}-probas.csv".format(runname, trial))
        probas_df.to_csv(probas_fn, index=False)
        print("level2: Wrote probas file: {}".format(probas_fn))

    
if __name__ == "__main__":
    main()
