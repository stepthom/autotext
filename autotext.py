import uuid
import sys
import argparse
import numpy as np

import json

import pandas as pd
import itertools
import os

import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import autosklearn.classification
from autosklearn.metrics import balanced_accuracy, precision, recall, f1
import ConfigSpace.read_and_write.json as config_json

from datetime import datetime


def get_metrics(y_true, y_pred):
    res = {}
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['f1'] = f1_score(y_true, y_pred, average="macro")
    res['recall'] = recall_score(y_true, y_pred, average="macro")
    res['precision'] = precision_score(y_true, y_pred, average="macro")
    return res

scorer = autosklearn.metrics.make_scorer(
    'f1_score',
    sklearn.metrics.f1_score
)

def do_autosklearn(config, features_train, y_train, features_val, y_val, features_test, y_test):
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
def do_lr(config, features_train, y_train, features_val, y_val, features_test, y_test):
    res = {}
    jobs = config.get('n_jobs', 1)

    scaler = StandardScaler()
    scaler = scaler.fit(features_train)
    features_train = scaler.transform(features_train)
    features_val = scaler.transform(features_val)
    if features_test is not None:
        features_test = scaler.transform(features_test)

    print("Running LR")
    pipe = LogisticRegressionCV(cv=5,
            Cs=100,
            max_iter=500,
            n_jobs=jobs,
            random_state=0).fit(features_train, y_train)
    print("... done fitting LR.")
    res['train_metrics'] = get_metrics(y_train, pipe.predict(features_train))
    res['val_metrics']   = get_metrics(y_val, pipe.predict(features_val))
    if features_test is not None and y_test is not None:
        y_pred_test = pipe.predict(features_test)
        res['test_metrics']   = get_metrics(y_test, y_pred_test)
        print('Test results:')
        print(classification_report(y_test, y_pred_test))

    return res

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
def do_gbc(config, features_train, y_train, features_val, y_val, features_test, y_test):
    res = {}
    jobs = config.get('n_jobs', 1)

    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,
            max_depth=1, random_state=0)
    param_grid = {
                  "max_depth": [5, 10],
                  "learning_rate": [0.07, 0.1, 0.15, 0.2, 0.3],
                  "n_estimators": [250]}

    print("Running GBC")
    pipe = GridSearchCV(clf, param_grid, cv=2, n_jobs=jobs, verbose=0)
    pipe = pipe.fit(features_train, y_train)
    print("... done fitting GBC.")

    res['best_params'] = pipe.best_params_


    res['train_metrics'] = get_metrics(y_train, pipe.predict(features_train))
    res['val_metrics']   = get_metrics(y_val, pipe.predict(features_val))
    if features_test is not None and y_test is not None:
        y_pred_test = pipe.predict(features_test)
        res['test_metrics']   = get_metrics(y_test, y_pred_test)
        print('Test results:')
        print(classification_report(y_test, y_pred_test))

    return res

import lightgbm as lgb
from lightgbm import LGBMClassifier
#from sklearn.model_selection import GridSearchCV
def do_lgbm(config, features_train, y_train, features_val, y_val, features_test, y_test):
    res = {}
    jobs = config.get('n_jobs', 1)

    clf = LGBMClassifier(silent=True, learning_rate=1.0, objective='binary', random_state=0)
    param_grid = {
                  "num_leaves": [31, 63, 100],
                  "max_depth": [5, 7, 15, None],
                  "learning_rate": [0.07, 0.866, 0.1, 0.12, 0.15, 0.2],
                  "n_estimators": [100, 250, 400, 500]}

    print("Running LightGBM")
    pipe = GridSearchCV(clf, param_grid, cv=2, n_jobs=jobs, verbose=1)
    pipe = pipe.fit(features_train, y_train,
        eval_metric='f1',
        eval_set=[(features_test, y_test)],
        early_stopping_rounds=50,
        verbose=False,
                        )
    print("... done fitting LightGBM.")

    res['best_params'] = pipe.best_params_

    res['train_metrics'] = get_metrics(y_train, pipe.predict(features_train))
    res['val_metrics']   = get_metrics(y_val, pipe.predict(features_val))
    if features_test is not None and y_test is not None:
        y_pred_test = pipe.predict(features_test)
        res['test_metrics']   = get_metrics(y_test, y_pred_test)
        print('Test results:')
        print(classification_report(y_test, y_pred_test))

    return res



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

    # This file will hold debugging output
    dumpfile = open('out/{}.log'.format(runname), "w")

    # Read the settings file
    with open(args.settings_file) as f:
        config = json.load(f)

    print("Read settings:", file=dumpfile)
    print(json.dumps(config, indent=4, sort_keys=True), file=dumpfile)

    results['settings'] = config

    # Define our Target
    target_col = config['target_col']

    # Quick sanity check
    for filenames in config.get('filenames', []):
        for filename in filenames:
            print("Checking existence of {}...".format(filename))
            if not os.path.isfile(filename):
                print("Does not exist! Exiting.")
                return 1

    i = 0
    for filenames in config.get('filenames', []):
        print("Reading features...")
        features_train = pd.read_csv(filenames[0])
        y_train = features_train.pop(target_col)
        features_val = pd.read_csv(filenames[1])
        y_val = features_val.pop(target_col)
        features_test = None
        y_test = None
        if (len(filenames) > 2):
            features_test = pd.read_csv(filenames[2])
            if target_col in features_test:
                y_test = features_test.pop(target_col)
        print("...done")

        results[i] = {}
        results[i]['filenames'] = filenames


        #results[i]['lr'] = do_lr(config, features_train, y_train, features_val, y_val, features_test, y_test)

        with open('out/{}-results.json'.format(runname), 'w') as fp:
            json.dump(results, fp, indent=4)

        results[i]['lgbm'] = do_lgbm(config, features_train, y_train, features_val, y_val, features_test, y_test)

        with open('out/{}-results.json'.format(runname), 'w') as fp:
            json.dump(results, fp, indent=4)

        #results[i]['gbc'] = do_gbc(config, features_train, y_train, features_val, y_val, features_test, y_test)

        with open('out/{}-results.json'.format(runname), 'w') as fp:
            json.dump(results, fp, indent=4)


        #results[i]['autosklearn'] = do_autosklearn(config, features_train, y_train, features_val, y_val, features_test, y_test)

        dumpfile.flush()
        with open('out/{}-results.json'.format(runname), 'w') as fp:
            json.dump(results, fp, indent=4)

        i = i + 1


if __name__ == "__main__":
    main()
