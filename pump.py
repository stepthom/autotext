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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper


from category_encoders.wrapper import PolynomialWrapper

import ConfigSpace.read_and_write.json as config_json

import datetime

import jsonpickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_metrics(y_true, y_pred):
    res = {}
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['f1'] = f1_score(y_true, y_pred, average="macro")
    res['recall'] = recall_score(y_true, y_pred, average="macro")
    res['precision'] = precision_score(y_true, y_pred, average="macro")
    res['report'] = classification_report(y_true, y_pred, output_dict=True)
    return res

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dump_results(runname, results):

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

    with open('out/{}-results.json'.format(runname), 'w') as fp:
        json.dump(results, fp, indent=4, cls=NumpyEncoder)


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


def custom_metric(X_test, y_test, estimator, labels, X_train, y_train,
                  weight_test=None, weight_train=None):
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X_test)
    test_loss = 1.0-accuracy_score(y_test, y_pred, sample_weight=weight_test)
    y_pred = estimator.predict(X_train)
    train_loss = 1.0-accuracy_score(y_train, y_pred, sample_weight=weight_train)
    alpha = 1.1
    print(test_loss * (1 + alpha) - alpha * train_loss, [test_loss, train_loss])
    return test_loss * (1 + alpha) - alpha * train_loss, [test_loss, train_loss]


def custom_metric1(X_test, y_test, estimator, labels, X_train, y_train,
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

def custom_metric2(X_test, y_test, estimator, labels, X_train, y_train,
                  weight_test=None, weight_train=None):
    from sklearn.metrics import log_loss
    y_pred = estimator.predict_proba(X_test)
    test_loss = log_loss(y_test, y_pred, labels=labels,
                         sample_weight=weight_test)
    y_pred = estimator.predict_proba(X_train)
    train_loss = log_loss(y_train, y_pred, labels=labels,
                          sample_weight=weight_train)
    alpha = 0.5
    return test_loss * (1 + alpha) - alpha * train_loss, [test_loss, train_loss]


def add_indicator(X, col, missing_val, replace_val=np.nan):
    col_new = "{}_missing".format(col)
    X[col_new] = X[col].apply(lambda x: 1 if x == missing_val else 0)
    X[col] = X[col].replace(missing_val, replace_val)
    return X

def add_datetime(X, column):
    tmp_dt = X[column].dt
    new_columns_dict = {f'year_{column}': tmp_dt.year, f'month_{column}': tmp_dt.month,
                        f'day_{column}': tmp_dt.day, f'hour_{column}': tmp_dt.hour,
                        f'minute_{column}': tmp_dt.minute, f'second_{column}': tmp_dt.second,
                        f'dayofweek_{column}': tmp_dt.dayofweek,
                        f'dayofyear_{column}': tmp_dt.dayofyear,
                        f'quarter_{column}': tmp_dt.quarter}
    for new_col_name in new_columns_dict.keys():
        if new_col_name not in X.columns and \
                new_columns_dict.get(new_col_name).nunique(dropna=False) >= 2:
            X[new_col_name] = new_columns_dict.get(new_col_name)
    return X

def get_top_n_values(X, column, start_list=[], n=5):

    top_values = start_list
  
    vc = X[column].value_counts(sort=True, ascending=False)
    vals = list(vc.index)
    if len(vals) > n:
        top_values = top_values + vals[0:n]
    else:
        top_values = top_values + vals
    return top_values

def keep_only(X, column, keep_list, replace_val='__OTHER__'):
    X.loc[~X[column].isin(keep_list), column] = replace_val
    return X



def hack(X, y=None, imputer=None, top_n_values=None, enc=None, train=False, keep_top=10):

    df = X.copy()

    ##################################################
    # Change Types
    ##################################################
    df['construction_year']= pd.to_numeric(df['construction_year'])

    ##################################################
    # Add missing value indicators
    ##################################################

    # TODO: replace with nan (so it will be imputed later?
    df = add_indicator(df, 'construction_year', 0, replace_val=1950)
    
    # TODO: replace with something else?
    df = add_indicator(df, 'amount_tsh', 0)

    # TODO: replace with region means?
    df = add_indicator(df, 'population', 0)
    df = add_indicator(df, 'latitude', 0)
    df = add_indicator(df, 'longitude', 0)
    df = add_indicator(df, 'gps_height', 0)

    ##################################################
    # Impute missing numeric values
    ##################################################
    # Impute Missing Value
    numeric_features = ['amount_tsh', 'gps_height', 
                        'longitude', 'latitude', 'num_private',
                        'population']

    if train:
        imputer = imputer.fit(df[numeric_features], y)
    df[numeric_features] = imputer.transform(df[numeric_features])

   
    ##################################################
    # Date/Time
    ##################################################
    df = add_datetime(df, 'date_recorded')

    baseline = pd.datetime(2014, 1, 1)
    df['date_recorded_since'] = (baseline - df['date_recorded']).dt.days

    df['timediff'] = df['year_date_recorded'] - df['construction_year']


    ##################################################
    # Categorical
    ##################################################
    cat_cols = []
    for c in df.columns:
        col_type = df[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            cat_cols.append(c)
            
    # Bools to strings
    df['public_meeting'] = df['public_meeting'].astype('str')
    df['permit'] = df['permit'].astype('str')

    # Convert np.nan and "none" to "__NAN__"
    df['wpt_name'] = df['wpt_name'].replace("none", '__NAN__')
    df[cat_cols] = df[cat_cols].fillna('__NAN__')
    
    
    # Categorical levels "smushing" - convert long-tail values to "__OTHER__"
    for c in ['wpt_name', 'funder', 'extraction_type', 'installer']:
        if train:
            top_n_values[c] = get_top_n_values(df, c, n=keep_top)
        df = keep_only(df, c, top_n_values[c])
        
    df[cat_cols] = df[cat_cols].astype('category')

    # Encoding
    if train:
        enc.fit(df[cat_cols], y)
    df[cat_cols] = enc.transform(df[cat_cols])
    df[cat_cols] = df[cat_cols].astype('category')

    ##################################################
    # Dropping
    ##################################################
    drop_cols = ['id', 'date_recorded', 'recorded_by']
    drop_cols = drop_cols + [ 'scheme_name']
    dup_cols = ['payment_type', 'quantity_group', 'source_type', 'waterpoint_type_group']
    drop_cols = drop_cols + dup_cols
    df = df.drop(drop_cols, axis=1)

    return df


def main():

    #parser = argparse.ArgumentParser()
    #parser.add_argument(
        #"settings_file", help="Path to JSON settings/config file.")
    #args = parser.parse_args()
    runname = str(uuid.uuid4())

    print("Run name: {}".format(runname))

    # This structure will hold all the results and will be dumped to disk.
    results = {}
    results['runname'] = runname
    results['starttime'] = str(datetime.datetime.now())
    results['hostname'] = socket.gethostname()

    #df = pd.read_csv("https://drive.google.com/uc?export=download&id=1O3gYw1FlsbDYrXhma5_N6AYqQ3OKI3uh", parse_dates=['date_recorded'])
    dfo = pd.read_csv("data/pump_train.csv", parse_dates=['date_recorded'])

    Xo = dfo.drop('status_group', axis=1)
    y = dfo['status_group']

    imputer = SimpleImputer(missing_values=np.nan, strategy="median")   
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    top_n_values = {}
    X = hack(Xo, y, imputer, top_n_values, enc, train=True, keep_top=10)
    
    if (False):

        params = {
         #'colsample_bytree': 0.8031986460435498,
         #'learning_rate': 0.13913418814669692,
         #'max_bin': 128,
         #'min_child_samples': 38,
         #'n_estimators': 74,
         'n_jobs': 4,
         #'num_leaves': 16,
         'objective': 'multiclass',
         #'reg_alpha': 0.0030439446132180708,
         #'reg_lambda': 0.2510232453366158,
         'subsample': 1.0}
        clf = lgb.LGBMClassifier(**params)

        pipe1 = Pipeline(steps=[('preprocessor', pre1),
                                ('clf', clf)])

        param_grid = {
            #'clf__colsample_bytree': [0.6, 0.7, 0.80, 0.9, 1],
            #'clf__max_depth': [-1, 10, 20, 30],
            #'preprocessor__ct__cat__encoder__randomized': [True, False],
            'clf__max_bin': [8, 16, 32, 64],
            'clf__num_leaves': [48, 64, 84, 128],
            #'clf__path_smooth': [0, 0.1],
            'clf__n_estimators': [500, 1000, 1250, 1500],
            'clf__learning_rate': [0.2, 0.225, 0.25, 0.275, 0.30, 0.325],
            'clf__class_weight':['balanced'],
        }


        search = GridSearchCV(pipe1, param_grid, n_jobs=20, cv=3, scoring='accuracy', return_train_score=True, verbose=2)

        print("Running GridSearchCV")
        search.fit(X, y)

        results['cv_results_'] = search.cv_results_
        tbl = cv_results_to_df(search.cv_results_)
        print(tbl)
        tbl.to_csv("out/tbl_{}.csv".format(runname), index=False)
                                  

    if True:
        
        pipe = AutoML()
        automl_settings = {
            "time_budget": 60,
            "task": 'classififcation',
            "log_file_name": "out/flaml-{}.log".format(runname),
            "n_jobs": 20,
            "estimator_list": ['lgbm', 'xgboost', 'rf', 'extra_tree'],
            "model_history": True,
            "eval_method": "cv",
            "n_splits": 5,
            "metric": 'accuracy',
            #"metric": custom_metric,
            "log_training_metric": True,
            "verbose": 1,
        }

        results['automl_settings'] = jsonpickle.encode(automl_settings, unpicklable=False, keys=True)

        results['starttime'] = str(datetime.datetime.now())
        
        pipe.fit(X, y,**automl_settings)
        results['endtime'] = str(datetime.datetime.now())

        results['best_estimator'] = pipe.best_estimator
        results['best_config'] = pipe.best_config
        results['best_loss'] = 1-pipe.best_loss
        results['best_model'] = '{}'.format(str(pipe.model))
        
        print(results['best_loss'])
        #print(results['metrics_test'])
        
        #test_df = pd.read_csv('https://drive.google.com/uc?export=download&id=1Qnrd0pIRHJNoqNXNEDfp4YJglF4mRL_6')
        test_df = pd.read_csv('data/pump_test.csv', parse_dates=['date_recorded'])
        _test = hack(test_df, None, imputer, top_n_values, enc, train=True, keep_top=10)
        
        
        preds = pipe.predict(_test)
        submission = pd.DataFrame(data={'id': test_df['id'], 'status_group': preds})
        submission.to_csv("out/{}-stepthom_submission.csv".format(runname), index=False)
        


    results['endtime'] = str(datetime.datetime.now())
    dump_results(runname, results)
    

    print("Run name: {}".format(runname))


if __name__ == "__main__":
    main()
