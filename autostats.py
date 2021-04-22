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
import os
import socket

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier

import datetime

import jsonpickle

from scipy import stats


def dump_results(fname, results):
    with open(fname, 'w') as fp:
        json.dump(results, fp, indent=4)


def get_numeric_stats(col):
    res = {}
    res = col.describe().to_dict()
    res['num_na'] = int(col.isna().sum())
    res['num_nonzero'] = int(col.astype(bool).sum())
    res['nunique'] = int(col.nunique())
    cf = stats.relfreq(col, numbins=10)
    res['relfreq'] = {}
    bins = [cf.lowerlimit]
    for i in range(1, 10+1):
        bins.append(bins[i-1] + cf.binsize)
    res['relfreq']['bins'] = bins
    res['relfreq']['frequency'] = cf.frequency.tolist()
    return res


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
    print("...done. Shape = {}".format(df.shape))

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

    print("Running ExtraTrees...")
    clf = ExtraTreesClassifier(n_estimators=50, random_state=42).fit(X, y)
    _scores = pd.DataFrame(
        {'col': X.columns, 'importance': clf.feature_importances_})
    _scores = _scores.sort_values(['importance'], ascending=False)
    ordered_cols = _scores['col']
    _scores = _scores.to_dict()
    results['feature_importances'] = _scores

    dump_results(args.output_file, results)

    # Correlation between columns; save for later
    print("Running Correlations...")
    corrs_p = df.corr(method='pearson').abs().unstack(
    ).sort_values(ascending=False).reset_index()
    corrs_p.columns = ['col1', 'col2', 'corr']
    corrs_s = df.corr(method='spearman').abs().unstack(
    ).sort_values(ascending=False).reset_index()
    corrs_s.columns = ['col1', 'col2', 'corr']
    corrs_k = df.corr(method='kendall').abs().unstack(
    ).sort_values(ascending=False).reset_index()
    corrs_k.columns = ['col1', 'col2', 'corr']

    feats = {}

    # for col_name in [col_name for col_name in ordered_cols]:
    rank = 0
    for row_idx, col_name in _scores['col'].items():
        rank = rank + 1
        print("row_idx: {}, col_name: {}, rank: {}".format(
            row_idx, col_name, rank))
        feats[col_name] = {}
        df[col_name] = df[col_name].replace({True: 1, False: 0})

        # Basic stats
        feats[col_name]['head'] = df[col_name].head().to_dict()
        feats[col_name]['5_largest'] = df[col_name].nlargest(5).to_dict()
        feats[col_name]['5_smallest'] = df[col_name].nsmallest(5).to_dict()

        if not is_numeric_dtype(df[col_name]):
            print("Warning: not implemented for non-numeric types.")
            continue

        feats[col_name]['basic_stats'] = get_numeric_stats(df[col_name])

        # Column-target statistics
        ts = {}
        ts['et_importance'] = _scores['importance'][row_idx]

        mi = mutual_info_classif(
            df[col_name].to_numpy().reshape(-1, 1), df[args.target_col])
        ts['mutual_info'] = mi.tolist()[0]

        ts['corr_pearson'] = stats.pearsonr(
            df[col_name].fillna(0), df[args.target_col])
        ts['corr_spearman'] = stats.spearmanr(
            df[col_name], df[args.target_col], nan_policy="omit")
        ts['linregress'] = stats.linregress(
            df[col_name].fillna(0), df[args.target_col])

        try:
            a = df[col_name][y == results['labels'][0]]
            b = df[col_name][y == results['labels'][1]]
            ts['mannwhitneyu'] = stats.mannwhitneyu(a, b)
        except ValueError:
            pass

        for label in results['labels']:
            _tmp = df[df[args.target_col] == label]
            ts['target_{}_stats'.format(label)] = get_numeric_stats(
                _tmp[col_name])

        feats[col_name]['target_stats'] = ts

        # Column-column statistics
        cs = {}
        cs['pearson'] = corrs_p.loc[corrs_p['col1'] ==
                                    col_name][['col2', 'corr']].head(10).to_dict()
        cs['kendall'] = corrs_k.loc[corrs_k['col1'] ==
                                    col_name][['col2', 'corr']].head(10).to_dict()
        cs['spearman'] = corrs_s.loc[corrs_s['col1'] ==
                                     col_name][['col2', 'corr']].head(10).to_dict()

        feats[col_name]['column_correlations'] = cs

    results['features'] = feats

    results['endtime'] = str(datetime.datetime.now())

    dump_results(args.output_file, results)


if __name__ == "__main__":
    main()
