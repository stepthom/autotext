from sklearn.linear_model import LogisticRegressionCV
from flaml import AutoML
from sklearn.preprocessing import StandardScaler
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

import jsonpickle



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


def custom_metric(X_test, y_test, estimator, labels, X_train, y_train,
                  weight_test=None, weight_train=None):
    from sklearn.metrics import f1_score
    y_pred = estimator.predict(X_test)
    test_loss = 1.0-f1_score(y_test, y_pred, labels=labels,
                         sample_weight=weight_test, average='macro')
    y_pred = estimator.predict(X_train)
    train_loss = 1.0-f1_score(y_train, y_pred, labels=labels,
                          sample_weight=weight_train, average='macro')
    alpha = 0.1
    return test_loss * (1 + alpha) - alpha * train_loss, [test_loss, train_loss]


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

    res['automl_settings'] = jsonpickle.encode(automl_settings, unpicklable=False, keys=True)

    print("Running FLAML with {} jobs for {} seconds...".format(jobs, time))
    res['starttime'] = str(datetime.datetime.now())
    pipe.fit(datasets.X_train, datasets.y_train, X_val=datasets.X_val,
             y_val=datasets.y_val, **automl_settings)
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

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

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
    X_train, y_train = read_and_split(
        train_fn, target_col, index_col, drop_cols)

    if test_fn is not None:
        print("Reading test file...")
        X_test, y_test = read_and_split(
            test_fn, target_col, index_col, drop_cols)
    else:
        print("Creating test data from training...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=0.1, random_state=1)

    X_val = None
    y_val = None
    if val_fn is not None:
        print("Reading val file...")
        X_val, y_val = read_and_split(val_fn, target_col, index_col, drop_cols)
    else:
        if not combine_train_and_val:
            print("Creating val data from training data...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.1, random_state=1)

    if combine_train_and_val:
        if X_val is not None and y_val is not None:
            print("Combining train and val datasets...")
            X_train = X_train.append(X_val)
            y_train = y_train.append(y_val)
            X_val = None
            y_val = None
        else:
            print("No val dataset to combine...")

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

    train_fn = config.get('train_filename', None)
    val_fn = config.get('val_filename', None)
    test_fn = config.get('test_filename', None)
    index_col = config.get('index_col', None)
    target_col = config.get('target_col', None)
    drop_cols = config.get('drop_cols', [])
    combine_train_and_val = config.get('combine_train_and_val', False)

    # Make DataSets
    datasets = read_and_split_all(
        train_fn, val_fn, test_fn, target_col, index_col, drop_cols, combine_train_and_val)

    results['X_train_shape'] = datasets.X_train.shape
    results['y_train_shape'] = datasets.y_train.shape
    if datasets.X_val is not None:
        results['X_val_shape'] = datasets.X_val.shape
        results['y_val_shape'] = datasets.y_val.shape
    results['X_test_shape'] = datasets.X_test.shape
    results['y_test_shape'] = datasets.y_test.shape

    results['X_train_head'] = datasets.X_train.head().to_dict()
    results['y_train_head'] = datasets.y_train.head().to_dict()
    results['X_test_head'] = datasets.X_test.head().to_dict()
    results['y_test_head'] = datasets.y_test.head().to_dict()

    dump_results(runname, results)

    pipe = Pipeline



    results['endtime'] = str(datetime.datetime.now())

    dump_results(runname, results)
    print("Run name: {}".format(runname))


if __name__ == "__main__":
    main()
