import uuid
import sys
import argparse
import numpy as np

import json

import pandas as pd
import itertools
import os
import socket

import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import autosklearn.classification
from autosklearn.metrics import balanced_accuracy, precision, recall, f1
import ConfigSpace.read_and_write.json as config_json

import datetime


def get_metrics(y_true, y_pred):
    res = {}
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['f1'] = f1_score(y_true, y_pred, average="macro")
    res['recall'] = recall_score(y_true, y_pred, average="macro")
    res['precision'] = precision_score(y_true, y_pred, average="macro")
    res['report'] = classification_report(y_true, y_pred, output_dict=True)
    return res

scorer = autosklearn.metrics.make_scorer(
    'f1_score',
    sklearn.metrics.f1_score
)

def do_autosklearn(config, features_train, y_train, ffeatures_test, y_test):
    res = {}

    time = config.get('time_left_for_this_task', 100)
    jobs = config.get('n_jobs', 1)

    print("Running autosklearn for {} seconds on {} jobs...".format(time, jobs))

    pipe = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=time,
        metric=scorer,
        n_jobs=jobs,
        seed=42,
        memory_limit=9072,
        exclude_estimators=config['exclude_estimators'],
    )

    pipe = pipe.fit(features_train, y_train)

    print("... done fitting autosklearn.")

    y_pred = pipe.predict(features_val)
    res['val_metrics'] = get_metrics(y_val, y_pred)

    if features_test is not None:
        y_pred_test = pipe.predict(features_test)
        #features_test['pred'] = y_pred_test
        #features_test.to_csv('out/{}-test_pred.csv'.format(), index=False)
        if y_test is not None:
            res['test_metrics'] = get_metrics(
                y_test, y_pred_test)
            print('Test results:')
            print(classification_report(y_test, y_pred_test))

    cv = pipe.cv_results_
    cv_df = pd.DataFrame.from_dict(cv)
    #cv_df.to_csv('out/{}-cv.csv'.format(runname), index=False)
    res['cv_results'] = {}
    for index, row in cv_df.iterrows():
        res['cv_results'][index] = {}
        res['cv_results'][index]['mean_test_score'] = row['mean_test_score']
        res['cv_results'][index]['mean_fit_time'] = row['mean_fit_time']
        res['cv_results'][index]['rank_test_scores'] = row['rank_test_scores']
        res['cv_results'][index]['status'] = row['status']
        res['cv_results'][index]['params'] = row['params']

    return res

from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
def do_lr(runname, config, datasets):
    res = {}
    jobs = config.get('n_jobs', 1)

    scaler = StandardScaler()
    scaler = scaler.fit(datasets.X_train)
    _X_train = scaler.transform(datasets.X_train)

    _X_val = None
    if (datasets.X_val is not None):
        _X_val = scaler.transform(datasets.X_val)
    _X_test = scaler.transform(datasets.X_test)

    pipe = LogisticRegressionCV(cv=5,
            Cs=100,
            max_iter=5000,
            n_jobs=jobs,
            random_state=42)

    print("Running LR")
    res['starttime'] = str(datetime.datetime.now())
    pipe = pipe.fit(datasets.X_train, datasets.y_train)
    res['endtime'] = str(datetime.datetime.now())
    print("... done fitting LR.")

    _preds_train = pipe.predict(_X_train)
    res['metrics_train'] = get_metrics(datasets.y_train, _preds_train)

    if (_X_val is not None):
        _preds_val = pipe.predict(_X_val)
        res['metrics_val'] = get_metrics(datasets.y_val, _preds_val)

    _preds_test = pipe.predict(_X_test)
    if datasets.y_test is not None:
        res['metrics_test'] = get_metrics(datasets.y_test, _preds_test)

    output_preds = config.get('output_preds', False)
    if output_preds:
        _preds_test_fn = 'out/{}-lr-preds.csv'.format(runname)
        res['preds_test_filename'] = _preds_test_fn
        _preds_df = pd.DataFrame(_preds_test, columns=['y_preds'])
        cols = datasets.X_test.columns.tolist() + ['y_test',  'y_pred']
        _df_test = pd.concat([datasets.X_test.reset_index(drop=True),
            datasets.y_test.reset_index(drop=True), _preds_df.reset_index(drop=True)], axis=1, ignore_index=True)
        _df_test.columns = cols
        _df_test.to_csv(_preds_test_fn, index=False)

    return datasets, res


from flaml import AutoML
def do_flaml(runname, config, datasets):
    res = {}
    time = config.get('time_left_for_this_task', 100)
    jobs = config.get('n_jobs', 1)

    pipe = AutoML()
    automl_settings = {
            "time_budget": time,  # in seconds
            "metric": 'f1',
            "task": 'classififcation',
            "log_file_name": "out/flaml-{}.log".format(runname),
            "n_jobs": jobs,
            "estimator_list": ['lgbm', 'xgboost', 'catboost', 'extra_tree'],
            "model_history": True,
        }

    res['automl_settings'] = automl_settings

    print("Running FLAML with {} jobs for {} seconds...".format(jobs, time))
    res['starttime'] = str(datetime.datetime.now())
    pipe.fit(datasets.X_train, datasets.y_train, X_val=datasets.X_test, y_val=datasets.y_test, **automl_settings)
    res['endtime'] = str(datetime.datetime.now())
    print("... done running FLAML.")

    res['best_estimator'] = pipe.best_estimator
    res['best_config'] = pipe.best_config
    res['best_f1_val'] = 1-pipe.best_loss
    res['best_model'] = '{}'.format(str(pipe.model))

    _preds_train = pipe.predict(datasets.X_train)
    res['metrics_train'] = get_metrics(datasets.y_train, _preds_train)

    if (datasets.X_val is not None) and (datasets.y_val is not None):
        _preds_val = pipe.predict(datasets.X_val)
        res['metrics_val'] = get_metrics(datasets.y_val, _preds_val)

    _preds_test = pipe.predict(datasets.X_test)
    if datasets.y_test is not None:
        res['metrics_test'] = get_metrics(datasets.y_test, _preds_test)

    output_preds = config.get('output_preds', False)
    if output_preds:
        _preds_test_fn = 'out/{}-flaml-preds.csv'.format(runname)
        res['preds_test_filename'] = _preds_test_fn
        _preds_df = pd.DataFrame(_preds_test, columns=['y_preds'])
        cols = datasets.X_test.columns.tolist() + ['y_test',  'y_pred']
        _df_test = pd.concat([datasets.X_test.reset_index(drop=True),
            datasets.y_test.reset_index(drop=True), _preds_df.reset_index(drop=True)], axis=1, ignore_index=True)
        _df_test.columns = cols
        _df_test.to_csv(_preds_test_fn, index=False)

    return datasets, res

def dump_results(runname, results):
    with open('out/{}-results.json'.format(runname), 'w') as fp:
        json.dump(results, fp, indent=4)


class DataSets:
    def __init__(self, train_fn, val_fn, test_fn,
           X_train,
           y_train,
           X_val,
           y_val,
           X_test,
           y_test
            ):
        self.train_fn = train_fn
        self.val_fn = val_fn
        self.test_fn = test_fn

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

def read_and_split(fn, target_col, index_col=None, drop_cols=[]):
    _df = pd.read_csv(fn, index_col=index_col)

    for drop_col in drop_cols:
        _df = _df[_df.columns.drop(list(_df.filter(regex=drop_col)))]

    # For safety: FLAML doens't like these kinds of chars in column names
    _df.columns = _df.columns.str.replace("[(),:)]", "_", regex=True)

    _X = _df.drop([target_col], axis=1)
    _y = None
    if target_col in _df:
        _y = _df[target_col]
    return _X, _y


def read_and_split_all(train_fn, val_fn, test_fn, target_col, index_col=None, drop_cols=[], combine_train_and_val=False):
    if not train_fn:
        raise ValueError('train_fn cannot be null.')

    print("Reading train file...")
    X_train, y_train = read_and_split(train_fn, target_col, index_col, drop_cols)

    if test_fn is not None:
        print("Reading test file...")
        X_test, y_test = read_and_split(test_fn, target_col, index_col, drop_cols)
    else:
        print("Creating test data from training...")
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

    if val_fn is not None:
        print("Reading val file...")
        X_val, y_val = read_and_split(val_fn, target_col, index_col, drop_cols)
    else:
        print("Creating val data from training data...")
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

    if combine_train_and_val:
        print("Combining train and val datasets...")
        X_train = X_train.append(X_val)
        y_train = y_train.append(y_val)
        X_val = None
        y_val = None

    return DataSets(train_fn, val_fn, test_fn,
             X_train, y_train,
             X_val, y_val,
             X_test, y_test)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "settings_file", help="Path to JSON settings/config file.")
    args = parser.parse_args()
    runname = str(uuid.uuid4())

    print("Run name: {}".format(runname))

    # This structure will hold all the results and will be dumped to disk.
    results = {}
    results['runname'] = runname
    results['starttime'] = str(datetime.datetime.now())
    results['hostname'] = socket.gethostname()

    # Read the settings file
    with open(args.settings_file) as f:
        config = json.load(f)

    results['settings'] = config

    train_fn   = config.get('train_filename', None)
    val_fn     = config.get('val_filename', None)
    test_fn    = config.get('test_filename', None)
    index_col  = config.get('index_col', None)
    target_col = config.get('target_col', None)
    drop_cols  = config.get('drop_cols', [])
    combine_train_and_val  = config.get('combine_train_and_val', False)

    # Make DataSets
    datasets = read_and_split_all(train_fn, val_fn, test_fn, target_col, index_col, drop_cols, combine_train_and_val)

    results['X_train_shape'] = datasets.X_train.shape
    results['y_train_shape'] = datasets.y_train.shape
    if datasets.X_val is not None:
        results['X_val_shape'] = datasets.X_val.shape
        results['y_val_shape'] = datasets.y_val.shape
    results['X_test_shape'] = datasets.X_test.shape
    results['y_test_shape'] = datasets.y_test.shape
    results['X_train_columns'] = datasets.X_train.columns.tolist()

    datasets, results['lr'] = do_lr(runname, config, datasets)
    dump_results(runname, results)

    datasets, results['flaml'] = do_flaml(runname, config, datasets)
    dump_results(runname, results)

    #results['autosklearn'] = do_autosklearn(config, features_train, y_train, features_test, y_test)

    results['endtime'] = str(datetime.datetime.now())

    dump_results(runname, results)
    print("Run name: {}".format(runname))



if __name__ == "__main__":
    main()
