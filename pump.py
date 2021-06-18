from sklearn.linear_model import LogisticRegressionCV
from flaml import AutoML
import uuid
import sys
import argparse
import numpy as np

import json

import pandas as pd
import itertools
import os
import socket

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import date, datetime
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
import category_encoders as ce
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

from category_encoders.wrapper import PolynomialWrapper

import ConfigSpace.read_and_write.json as config_json

import datetime

import jsonpickle


def get_metrics(y_true, y_pred):
    res = {}
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['f1'] = f1_score(y_true, y_pred, average="macro")
    res['recall'] = recall_score(y_true, y_pred, average="macro")
    res['precision'] = precision_score(y_true, y_pred, average="macro")
    res['report'] = classification_report(y_true, y_pred, output_dict=True)
    return res

def dump_results(runname, results):

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

    with open('out/{}-results.json'.format(runname), 'w') as fp:
        json.dump(results, fp, indent=4)


# Helper function to print out the results of hyperparmater tuning in a nice table.
def cv_results_to_df(cv_results):
    results = pd.DataFrame(list(cv_results['params']))
    results['mean_fit_time'] = cv_results['mean_fit_time']
    results['mean_score_time'] = cv_results['mean_score_time']
    results['mean_train_score'] = cv_results['mean_train_score']
    results['std_train_score'] = cv_results['std_train_score']
    results['mean_test_score'] = cv_results['mean_test_score']
    results['std_test_score'] = cv_results['std_test_score']
    results['rank_test_score'] = cv_results['rank_test_score']

    results = results.sort_values(['mean_test_score'], ascending=False)
    return results

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

    df = pd.read_csv("https://drive.google.com/uc?export=download&id=1O3gYw1FlsbDYrXhma5_N6AYqQ3OKI3uh")
    df.info()
    
    X = df.drop('status_group', axis=1)
    y = df['status_group']
 

    drop_cols = ['id', 'date_recorded', 'subvillage', 'recorded_by']
    drop_cols = drop_cols + ['region_code', 'scheme_name']

    dup_cols = ['payment_type', 'quantity_group', 'source_type', 'waterpoint_type_group']

    drop_cols = drop_cols + dup_cols

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.drop(drop_cols, errors='ignore')
    categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns.drop(drop_cols, errors='ignore')

    # Fix-zeroes:
    # Construction year
    # amount_tsh

   
    categorical_transformer = Pipeline(steps=[
        ('encoder', ce.wrapper.PolynomialWrapper(ce.m_estimate.MEstimateEncoder(randomized=True, verbose=2))),
        ])

    numeric_transformer = Pipeline(steps=[
        (('imputer'), SimpleImputer(missing_values=np.nan, strategy="median")),                                  
        ('scaler', StandardScaler()),
        ])


    pre1 = Pipeline(steps=[
        ('ct', ColumnTransformer(
          transformers=[
              ('num', numeric_transformer, numeric_features),
              ('cat', categorical_transformer, categorical_features),
              ('drop', 'drop', drop_cols)],
              remainder = 'passthrough', sparse_threshold=0)),
      ])

    param_grid = {
        'clf__colsample_bytree': [0.80, 1],
        'clf__max_depth': [-1, 10],
        #'clf__n_estimators': [50, 75, 150, 300], 
        'clf__class_weight':['balanced'],
    }

    params = {
     #'colsample_bytree': 0.8031986460435498,
     'learning_rate': 0.13913418814669692,
     'max_bin': 128,
     'min_child_samples': 38,
     'n_estimators': 74,
     'n_jobs': 4,
     'num_leaves': 16,
     'objective': 'multiclass',
     'reg_alpha': 0.0030439446132180708,
     'reg_lambda': 0.2510232453366158,
     'subsample': 1.0}
    clf = lgb.LGBMClassifier(**params)

    pipe1 = Pipeline(steps=[('preprocessor', pre1),  
                            ('clf', clf)])

    search = GridSearchCV(pipe1, param_grid, cv=5, scoring='accuracy', return_train_score=True, verbose=2)


    print("Running GridSearchCV")
    search.fit(X, y)
    
    
    results['cv_results_'] = search.cv_results_

    
    print(cv_results_to_df(search.cv_results_))
    
  
    results['endtime'] = str(datetime.datetime.now(
    dump_results(runname, results)
    print("Run name: {}".format(runname))


if __name__ == "__main__":
    main()
