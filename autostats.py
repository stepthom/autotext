from sklearn.linear_model import LogisticRegressionCV
from flaml import AutoML
from sklearn.preprocessing import StandardScaler
import uuid
import sys
import argparse
import numpy as np

import json

import pandas as pd
from pandas.api.types import is_numeric_dtype
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

from scipy import stats


def dump_results(fname, results):
    with open(fname, 'w') as fp:
        json.dump(results, fp, indent=4)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file", help="Path to data file on which to run stats.")
    parser.add_argument(
        "--target_col", help="Name of target column in data_file.")
    parser.add_argument(
        "--output_file", help="Path to output results file.", nargs="?", default="out/stats-out.json")
    parser.add_argument(
        "--drop_cols", help="Columns to drop.", nargs="*", default=[])
    args = parser.parse_args()


    # This structure will hold all the results and will be dumped to disk.
    results = {}
    results['starttime'] = str(datetime.datetime.now())
    results['hostname'] = socket.gethostname()

    print("Reading data file...")
    df = pd.read_csv(args.data_file)
    results['data_shape'] = df.shape
    print("Done. Shape = {}".format(df.shape))

    for drop_col in args.drop_cols:
        df = df[df.columns.drop(list(df.filter(regex=drop_col)))]

    has_nas = []
    results['na_cols'] = {}
    for col_name in [col_name for col_name in df.columns]:
        _sum = df[col_name].isna().sum()
        if _sum > 0:
            results['na_cols'][col_name] = int(_sum)

    df = df.dropna(axis=0, how='any')
    results['data_shape_after_dropna'] = df.shape

    results['labels'] = df[args.target_col].unique().tolist()

    X = df.drop([args.target_col], axis=1)
    y = df[args.target_col]

    N = 30
    from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
    s = SelectKBest(k='all')
    s.fit(X, y)
    _scores = pd.DataFrame(
        {'col': X.columns, 'score': s.scores_, 'pvalue': s.pvalues_})
    results['f_classif'] = _scores.sort_values(
        ['score'], ascending=False).head(N).to_dict()

    s = SelectKBest(mutual_info_classif, k='all')
    s.fit(X, y)
    _scores = pd.DataFrame({'col': X.columns, 'score': s.scores_})
    results['mutual_info_classif'] = _scores.sort_values(
        ['score'], ascending=False).head(N).to_dict()

    from sklearn.ensemble import ExtraTreesClassifier
    clf = ExtraTreesClassifier(n_estimators=50).fit(X, y)
    _scores = pd.DataFrame(
        {'col': X.columns, 'importance': clf.feature_importances_})
    results['extra_tree_importances'] = _scores.sort_values(
        ['importance'], ascending=False).head(N).to_dict()

    dump_results(args.output_file, results)

    for col_name in [col_name for col_name in X.columns]:
        print("col: {}".format(col_name))
        results[col_name] = {}
        df[col_name] = df[col_name].replace({True: 1, False: 0})
        results[col_name]['head'] = df[col_name].head().to_dict()


        def get_numeric_stats(col, tar):
            res = {}
            res['stats'] = col.describe().to_dict()
            res['num_na'] = int(col.isna().sum())
            res['num_nonzero'] = int(col.astype(bool).sum())
            res['nunique'] = int(col.nunique())
            res['5_largest'] = col.nlargest(5).to_dict()
            res['5_smallest'] = col.nsmallest(5).to_dict()

            return res

        if is_numeric_dtype(df[col_name]):
            results[col_name]['all'] = get_numeric_stats(
                df[col_name], df[args.target_col])
            for label in results['labels']:
                _tmp = df[df[args.target_col] == label]
                results[col_name]['target_{}'.format(label)] = get_numeric_stats(
                    _tmp[col_name], _tmp[args.target_col])

            try:
                a = df[col_name][y == results['labels'][0]]
                b = df[col_name][y == results['labels'][1]]
                results[col_name]['mannwhitneyu'] = stats.mannwhitneyu(a, b)
            except ValueError:
                pass

            results[col_name]['target_corr_pearson'] = stats.pearsonr(
                df[col_name].fillna(0), df[args.target_col])
            results[col_name]['target_corr_spearman'] = stats.spearmanr(
                df[col_name], df[args.target_col], nan_policy="omit")
            results[col_name]['target_linregress'] = stats.linregress(
                df[col_name].fillna(0), df[args.target_col])

    results['endtime'] = str(datetime.datetime.now())


    dump_results(args.output_file, results)


if __name__ == "__main__":
    main()
