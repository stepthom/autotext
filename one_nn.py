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
from sklearn.model_selection import train_test_split
import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
import time

import tensorflow as tf
import autokeras as ak

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Comet params
    parser.add_argument('--enable-comet', type=int, default=1)

    # Prep/FE params
    parser.add_argument('--cv', type=int, default=1)
    parser.add_argument('--sample-frac', type=float, default=1.0)


    args = parser.parse_args()

    print("Command line arguments:")
    print(vars(args))

    id_col = 'building_id'
    target_col = 'damage_grade'
    out_dir = 'earthquake/out'
    data_id = '000'
    geo_id_set = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

    # Create an experiment with your api key
    exp=None
    if args.enable_comet == 1:
        exp = Experiment(
            project_name="eq_one_nn",
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

    prep1 = Pipeline([
        ('typer', SteveFeatureTyper(cols=geo_id_set, typestr='category'))
    ])

    prep1.fit(X)
    X = prep1.transform(X)
    X_test = prep1.transform(X_test)

    cat_cols, num_cols, bin_cols, float_cols, date_cols = get_data_types(X, id_col, target_col)

    runname = ""
    if exp:
        runname = exp.get_key()
    else:
        runname = str(uuid.uuid4())
    run_fn = os.path.join(out_dir, "tune_eq_{}.json".format(runname))
    print("tune_eq: Run name: {}".format(runname))


    starttime = datetime.datetime.now()
    feat_imp = None

    if exp is not None:
        exp.log_other('train_fn', train_fn)
        exp.log_other('test_fn', test_fn)
        exp.log_table('X_head.csv', X.head())
        exp.log_parameters(vars(args))
        exp.log_asset('SteveHelpers.py')

    val_scores = []
    train_scores = []
    best_iterations = []
    probas = list()
    cv_step = 0
    if args.cv == 1:
        # Cross validation loop
        skf = StratifiedKFold(n_splits=12, random_state=12, shuffle=True)

        for train_index, val_index in skf.split(X, y):
            cv_step = cv_step+1

            X_train, X_val = X.loc[train_index], X.loc[val_index]
            y_train, y_val = y[train_index], y[val_index]

            start = time.time()
            print("ak train..")
            clf = ak.StructuredDataClassifier(overwrite=True, 
                                              max_trials=1,
                                             objective="val_loss")
            clf.fit(X_train, y_train, epochs=3)
            train_time = (time.time() - start)
            print("..done. {} secs".format(train_time))

            y_val_pred = clf.predict(X_val).astype(int)
            print("Val:")
            print(classification_report(y_val, y_val_pred, digits=4))

            custom_metrics = {
                #'val_log_loss':  log_loss(y_val, y_val_pred_proba),
                'val_micro_f1':  f1_score(y_val, y_val_pred, average="micro"),
                #'val_macro_f1':  f1_score(y_val, y_val_pred, average="macro"),
                #'val_weighted_f1':  f1_score(y_val, y_val_pred, average="weighted"),
                #'val_roc_auc':  roc_auc_score(y_val, y_val_pred_proba, multi_class="ovo"),
            }

            if exp is not None:
                exp.log_confusion_matrix(y_val, y_val_pred, title="Val Confusion Matrix {}".format(cv_step))
                
            y_train_pred = clf.predict(X_train).astype(int)

            custom_metrics.update({
                #'train_log_loss':  log_loss(y_train, y_train_pred_proba),
                'train_micro_f1':  f1_score(y_train, y_train_pred, average="micro"),
                #'train_macro_f1':  f1_score(y_train, y_train_pred, average="macro"),
                #'train_weighted_f1':  f1_score(y_train, y_train_pred, average="weighted"),
                #'train_roc_auc':  roc_auc_score(y_train, y_train_pred_proba, multi_class="ovo"),
            })
            print("Train:")
            print(classification_report(y_train, y_train_pred, digits=4))

            custom_metrics.update({
                "train_seconds": train_time,
                "cv_step": cv_step,
            })

            print("Fold metrics:")
            print(custom_metrics)

            val_scores.append(custom_metrics['val_micro_f1'])
            train_scores.append(custom_metrics['train_micro_f1'])

            if exp is not None:
                exp.log_metrics(custom_metrics, step=cv_step)
                exp.log_metric("mean_val_score", np.mean(val_scores), step=cv_step)
                exp.log_metric("mean_train_score", np.mean(train_scores), step=cv_step)
                exp.log_metric("mean_best_iteration", np.mean(best_iterations), step=cv_step)

        print("Train Scores: {} {}".format(np.mean(train_scores), train_scores))
        print("Val Scores: {} {}".format(np.mean(val_scores), val_scores))
        print("Best Iteration: {} {}".format(np.mean(best_iterations), best_iterations))
        if exp is not None:
            exp.log_metric("mean_val_score", np.mean(val_scores))
            exp.log_metric("mean_train_score", np.mean(train_scores))
            exp.log_metric("mean_best_iteration", np.mean(best_iterations))



    if len(probas) > 1:
        probas_df = pd.DataFrame(np.mean(probas, axis=0), columns=["1", "2", "3"])
        probas_df[id_col] = test_df[id_col]
        probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
        probas_fn = os.path.join(out_dir, "{}-{}-cv-probas.csv".format(runname, data_id))
        probas_df.to_csv(probas_fn, index=False)
        print("tune_eq: Wrote CV probas file: {}".format(probas_fn))


    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
    start = time.time()
    print("ak train..")
    clf = ak.StructuredDataClassifier(overwrite=True, 
                                      max_trials=50,
                                     objective="val_loss")
    clf.fit(X_train, y_train, epochs=20)
    train_time = (time.time() - start)
    print("..done. {} secs".format(train_time))

    preds = clf.predict(X_val).astype(int)
    print("Val:")
    print(classification_report(y_val, preds, digits=4))

    print("ak predict..")
    preds = clf.predict(X_test).astype(int)

    print("outputting files..")
    preds_df = pd.DataFrame(data={'id': test_df[id_col], target_col: preds[:, 0]})
    preds_fn = os.path.join(out_dir, "{}-{}-preds.csv".format(runname, data_id))
    preds_df.to_csv(preds_fn, index=False)
    print("tune_eq: Wrote preds file: {}".format(preds_fn))

if __name__ == "__main__":
    main()
