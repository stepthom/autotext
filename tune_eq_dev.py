import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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

from flaml import AutoML

from SteveHelpers import dump_json, get_data_types
from SteveHelpers import SteveCorexWrapper
from SteveHelpers import SteveNumericNormalizer
from SteveHelpers import SteveAutoFeatLight
from SteveHelpers import SteveFeatureDropper
from SteveHelpers import SteveFeatureTyper
from SteveHelpers import SteveEncoder
from SteveHelpers import SteveNumericCapper

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--enable-comet', type=int, default=1)
    parser.add_argument('--geo-id-set', type=int, default=3)
    parser.add_argument('--algo-set', type=int, default=1)
    parser.add_argument('--n-hidden', type=int, default=4)
    parser.add_argument('--dim-hidden', type=int, default=2)
    parser.add_argument('--smooth-marginals', type=int, default=0)
    parser.add_argument('--min-sample-leaf', type=int, default=5)
    parser.add_argument('--time-budget', type=int, default=60)
    parser.add_argument('--smoothing', type=float, default=10.0)
    parser.add_argument('--autofeat', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--ensemble', type=int, default=0)
    parser.add_argument('--metric', type=str, default="micro_f1")

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
            project_name="eq1",
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

    if False and args.n_hidden >= 1:
        steps.append(('corex', SteveCorexWrapper(bin_cols, n_hidden=args.n_hidden)))
        steps.append(('dropper', SteveFeatureDropper(bin_cols)))

    #steps.append(('cat_encoder', SteveEncoder(
        #cols=list(set(cat_cols) - set(geo_id_set)),
        #encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32))))

    steps.append(('num_capper', SteveNumericCapper(num_cols=['age'], max_val=30)))

    if args.autofeat == 1:
        steps.append(('num_autofeat', SteveAutoFeatLight(float_cols)))

    if args.normalize == 1:
        steps.append(('num_normalizer', SteveNumericNormalizer(float_cols, drop_orig=True)))

    print(steps)

    preprocessor = Pipeline(steps)

    preprocessor.fit(X)
    X = preprocessor.transform(X)
    X_test = preprocessor.transform(X_test)

    print("X head:")
    print(X.head().T)
    print("X dtypes:")
    print(X.dtypes)
    print("X_test head:")
    print(X_test.head().T)

    results = {}
   
    runname = ""
    if exp:
        runname = exp.get_key()
    else:
        runname = str(uuid.uuid4())
    run_fn = os.path.join(out_dir, "tune_eq_{}.json".format(runname))
    print("tune_eq: Run name: {}".format(runname))

    os.environ['OS_STEVE_MIN_SAMPLE_LEAF'] = str(args.min_sample_leaf)
    os.environ['OS_STEVE_SMOOTHING'] = str(args.smoothing)

    results['runname'] = runname
    results['args'] = vars(args)
    
    starttime = datetime.datetime.now()

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
   
    # Log some things before fit(), to more-easily monitor runs
    if exp is not None:
        exp.log_other('train_fn', train_fn)
        exp.log_other('test_fn', test_fn)
        exp.log_table('X_head.csv', X.head())
        exp.log_table('y_head.csv', y.head())
        exp.log_parameters(vars(args))
        exp.log_parameters(automl_settings)
        
    clf = AutoML()
    clf.fit(X, y, **automl_config, **automl_settings)

    endtime = datetime.datetime.now()
    duration = (endtime - starttime).seconds
    #results['automl_settings'] =  automl_settings
    results['best_score'] =  1 - clf.best_loss
    results['best_config'] =  clf.best_config
    results['best_estimator'] =  str(clf.best_estimator)
    results['endtime'] = str(endtime)
   
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
   
    
    print("Best model")
    bm = clf.best_model_for_estimator(clf.best_estimator)
    print(type(bm))
    print(bm)
    print(bm.estimator_class)
    print(bm.model)
    feature_names = bm.feature_names_
    feat_imp =  pd.DataFrame({'Feature': feature_names, 'Importance': bm.model.feature_importances_}).sort_values('Importance', ascending=False)
    print("Feature importances")
    print(feat_imp.head())
    
    if exp is not None:
        exp.log_metric("duration", duration)
        exp.log_metric("best_config_train_time", clf.best_config_train_time)
        exp.log_table('feat_imp.csv', feat_imp)

        #exp.log_metric(name="best_estimator", value=clf.best_estimator)
        #exp.log_asset_data(clf.best_config, name="best_config")
        exp.log_text(clf.best_config)

if __name__ == "__main__":
    main()
