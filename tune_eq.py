import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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

from flaml import AutoML

from SteveHelpers import dump_json, get_data_types
from SteveHelpers import SteveCorexWrapper
from SteveHelpers import SteveNumericNormalizer
from SteveHelpers import SteveAutoFeatLight
from SteveHelpers import SteveFeatureDropper
from SteveHelpers import SteveFeatureTyper

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--geo-id-set', type=int, default=3)
    parser.add_argument('-a', '--algo-set', type=int, default=1)
    parser.add_argument('--n-hidden', type=int, default=4)
    parser.add_argument('--dim-hidden', type=int, default=2)
    parser.add_argument('--smooth-marginals', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--min-sample-leaf', type=int, default=1)
    parser.add_argument('--smoothing', type=float, default=10.0)
    parser.add_argument('--autofeat', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--normalize', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--ensemble', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    
    args = parser.parse_args()
    
    print(args)
    
    id_col = 'building_id'
    target_col = 'damage_grade'
    out_dir = 'earthquake/out'
    data_id = '000'
    metric = "micro_f1"

    train_df  = pd.read_csv('earthquake/earthquake_train.csv')
    #train_df  = train_df.sample(frac=0.1, random_state=3).reset_index(drop=True)
    test_df  = pd.read_csv('earthquake/earthquake_test.csv')
    
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
        estimator_list = ['lgbm', 'xgboost', 'catboost']
    elif args.algo_set == 4:
        estimator_list = ['catboost']

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
    steps.append(('corex', SteveCorexWrapper(bin_cols, n_hidden=args.n_hidden)))
    steps.append(('dropper', SteveFeatureDropper(bin_cols)))
    
    if args.autofeat:
        steps.append(('num_autfeat', SteveAutoFeatLight(float_cols)))
        
    if args.normalize:
        steps.append(('num_normalizer', SteveNumericNormalizer(float_cols, drop_orig=True)))
    
    preprocessor = Pipeline(steps)
    
    preprocessor.fit(X)
    X = preprocessor.transform(X)
    X_test = preprocessor.transform(X_test)
   
    print("X head:", file=sys.stderr)
    print(X.head().T, file=sys.stderr)
    print("X dtypes:", file=sys.stderr)
    print(X.dtypes, file=sys.stderr)
    print("X_test head:", file=sys.stderr)
    print(X_test.head().T, file=sys.stderr)
   
    results = {}
    runname = str(uuid.uuid4())
    run_fn = os.path.join(out_dir, "tune_eq_{}.json".format(runname))
    print("tune_eq: Run name: {}".format(runname))
   
    os.environ['OS_STEVE_MIN_SAMPLE_LEAF'] = str(args.min_sample_leaf)
    os.environ['OS_STEVE_SMOOTHING'] = str(args.smoothing)

    results['runname'] = runname
    results['args'] = vars(args)
    results['hostname'] = socket.gethostname()
    results['starttime'] = str(datetime.datetime.now())

    automl_settings = {
        "time_budget": 50000,
        "log_file_name": "logs/flaml-{}.log".format(runname),
        "task": 'classification',
        "n_jobs": 8,
        "estimator_list": estimator_list,
        "model_history": False,
        "eval_method": "cv",
        "n_splits": 3,
        "metric": metric,
        "log_training_metric": True,
        "verbose": 1,
        "ensemble": args.ensemble,
    }
    clf = AutoML()
    clf.fit(X, y, **automl_settings)

    endtime = str(datetime.datetime.now())
    results['endtime'] = endtime
    results['automl_settings'] =  automl_settings
    results['best_score'] =  1 - clf.best_loss
    results['best_config'] =  clf.best_config
    results['best_estimator'] =  clf.best_estimator
    
    print("Run name: {}".format(runname))
    print("Run file name: {}".format(run_fn))
   
    preds = clf.predict(X_test)
    preds_df = pd.DataFrame(data={'id': test_df[id_col], target_col: preds})
    preds_fn = os.path.join(out_dir, "{}-{}-preds.csv".format(runname, data_id))
    preds_df.to_csv(preds_fn, index=False)
    print("tune_eq: Wrote preds file: {}".format(preds_fn))

    probas = clf.predict_proba(X_test)
    columns = None
    if hasattr(clf, 'classes_'):
        columns = clf.classes_
    probas_df = pd.DataFrame(probas, columns=columns)
    probas_df[id_col] = test_df[id_col]
    probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
    probas_fn = os.path.join(out_dir, "{}-{}-probas.csv".format(runname, data_id))
    probas_df.to_csv(probas_fn, index=False)
    print("tune_eq: Wrote probas file: {}".format(probas_fn))

    results['preds_fn'] = preds_fn
    results['probas_fn'] = probas_fn
    dump_json(run_fn, results)

if __name__ == "__main__":
    main()
