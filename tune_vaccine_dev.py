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
from sklearn.impute import SimpleImputer

from flaml import AutoML

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
    parser.add_argument('--enable-comet', type=int, default=1)
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--ensemble', type=int, default=0)
    parser.add_argument('--run-type', type=str, default="flaml")
    parser.add_argument('--metric', type=str, default="roc_auc")
    parser.add_argument('--time-budget', type=int, default=60)
    parser.add_argument('--target-col', type=str, default="h1n1_vaccine")
    
    parser.add_argument('--corex', type=int, default=-1)
    
    parser.add_argument('--cat-encoder', type=int, default=1)
    
    parser.add_argument('--min-sample-leaf', type=int, default=5)
    parser.add_argument('--smoothing', type=float, default=10.0)
    
    parser.add_argument('--autofeat', type=int, default=0)
    
    parser.add_argument('--normalize', type=int, default=0)
   

    parser.add_argument('--algo-set', type=int, default=1)
    

    args = parser.parse_args()

    print("Command line arguments:")
    print(vars(args))

    id_col = 'respondent_id'
    target_col = 'h1n1_vaccine'
    out_dir = 'h1n1/out'
    data_id = '000'
    train_fn = "h1n1/vaccine_h1n1_train.csv"
    test_fn = "h1n1/vaccine_h1n1_test.csv"
    
    if args.target_col == "seasonal_vaccine":
        id_col = 'respondent_id'
        target_col = 'seasonal_vaccine'
        out_dir = 'seasonal/out'
        data_id = '000'
        train_fn = "seasonal/vaccine_seasonal_train.csv"
        test_fn = "seasonal/vaccine_seasonal_test.csv"
        

    # Create an experiment with your api key
    exp=None
    if args.enable_comet == 1:
        exp = Experiment(
            project_name=args.target_col,
            workspace="stepthom",
            parse_args=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            log_graph=False
        )
    
    train_df  = pd.read_csv(train_fn)
    test_df  = pd.read_csv(test_fn)
    if args.sample_frac < 1.0:
        train_df  = train_df.sample(frac=args.sample_frac, random_state=3).reset_index(drop=True)
    

    estimator_list = ['lgbm']
    if args.algo_set == 1:
        estimator_list = ['lgbm']
    elif args.algo_set == 2:
        estimator_list = ['xgboost']
    elif args.algo_set == 3:
        estimator_list = ['catboost']
    elif args.algo_set == 4:
        estimator_list = ['rf']

    X = train_df.drop([id_col, target_col], axis=1)
    y = train_df[target_col]
    X_test = test_df.drop([id_col], axis=1)

    cat_cols, num_cols, bin_cols, float_cols, date_cols = get_data_types(X, id_col, target_col)

    steps = []
   
    steps.append(('missing_indicator', SteveMissingIndicator(num_cols)))
    steps.append(('num_imputer', SteveNumericImputer(num_cols,  SimpleImputer(missing_values=np.nan, strategy="median"))))
    
    if args.corex >= 1:
        steps.append(('typer', SteveFeatureTyper(bin_cols, typestr='int64')))
        steps.append(('corex', SteveCorexWrapper([col for col in bin_cols if "_missing" not in col], n_hidden=args.corex)))
        steps.append(('dropper', SteveFeatureDropper(bin_cols)))
        #steps.append(('dropper', SteveFeatureDropper(like='_corex', inverse=True)))
        
    steps.append(('cat_impute', SteveCategoryImputer(cat_cols)))
    steps.append(('cat_smush', SteveCategoryCoalescer(keep_top=25, cat_cols=cat_cols)))
       
    if args.cat_encoder == 1:
        pass
    elif args.cat_encoder == 2:
        steps.append(('cat_encoder', SteveEncoder(cols=cat_cols, 
                      encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32))))
        steps.append(('dropper', SteveFeatureDropper(cat_cols)))
    elif args.cat_encoder == 3:
        steps.append(('cat_encoder', SteveEncoder(cols=cat_cols,
                       encoder=OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int32))))
        steps.append(('dropper', SteveFeatureDropper(cat_cols)))
    
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
    
    print(list(X.columns))

    runname = ""
    if exp:
        runname = exp.get_key()
    else:
        runname = str(uuid.uuid4())
    run_fn = os.path.join(out_dir, "tune_eq_{}.json".format(runname))
    print("tune_eq: Run name: {}".format(runname))
    
    
    if exp is not None:
        exp.log_other('train_fn', train_fn)
        exp.log_other('test_fn', test_fn)
        exp.log_table('X_head.csv', X.head())
        exp.log_table('y_head.csv', y.head())
        exp.log_parameters(vars(args))

    starttime = datetime.datetime.now()
    feat_imp = None
    if args.run_type == "flaml":
        os.environ['OS_STEVE_MIN_SAMPLE_LEAF'] = str(args.min_sample_leaf)
        os.environ['OS_STEVE_SMOOTHING'] = str(args.smoothing)

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
            exp.log_parameters(automl_settings)

        clf = AutoML()
        clf.fit(X, y, **automl_config, **automl_settings)
        best_loss = clf.best_loss
        
        print("Best model")
        bm = clf.best_model_for_estimator(clf.best_estimator)
        print(bm.model)
        feature_names = bm.feature_names_
        feat_imp =  pd.DataFrame({'Feature': feature_names, 'Importance': bm.model.feature_importances_}).sort_values('Importance', ascending=False)
        print("Feature importances")
        print(feat_imp.head())
        
    else:
        pass

    endtime = datetime.datetime.now()
    duration = (endtime - starttime).seconds

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

    results = {}
    results['runname'] = runname
    results['preds_fn'] = preds_fn
    results['probas_fn'] = probas_fn
    results['endtime'] = str(endtime)
    results['algo_set'] = args.algo_set
    results['best_loss'] =  best_loss
    dump_json(run_fn, results)
   
    if exp is not None:
        exp.log_metric("duration", duration)
        
        if hasattr(clf, "best_config_train_train"):
            exp.log_metric("best_config_train_time", clf.best_config_train_time)
        
        if feat_imp is not None:
            exp.log_table('feat_imp.csv', feat_imp)

        if hasattr(clf, "best_config"):
            exp.log_text(clf.best_config)
            
        if hasattr(clf, "best_estimator"):
            exp.log_text(clf.best_model_for_estimator(clf.best_estimator).model)

if __name__ == "__main__":
    main()
