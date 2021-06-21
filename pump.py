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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper


import ConfigSpace.read_and_write.json as config_json

import datetime

import jsonpickle

import geopy.distance

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
    # Dropping - Won't need at all
    ##################################################
    drop_cols = ['id', 'scheme_name', 'recorded_by']
    dup_cols = ['payment_type', 'quantity_group', 'source_type', 'waterpoint_type_group']
    drop_cols = drop_cols + dup_cols
    df = df.drop(drop_cols, axis=1)

    ##################################################
    # Change Types
    ##################################################
    df['construction_year']= pd.to_numeric(df['construction_year'])

    ##################################################
    # Add missing value indicators
    ##################################################

    # TODO: replace with nan (so it will be imputed later?
    df = add_indicator(df, 'construction_year', 0)
    
    # TODO: replace with something else?
    df = add_indicator(df, 'amount_tsh', 0)

    # TODO: replace with region means?
    #df = add_indicator(df, 'population', 0)
    df = add_indicator(df, 'latitude', 0)
    df = add_indicator(df, 'longitude', 0)
    df = add_indicator(df, 'gps_height', 0)

    ##################################################
    # Impute missing numeric values
    ##################################################
    # Impute Missing Value
    numeric_features = ['amount_tsh', 'gps_height', 
                        'longitude', 'latitude', 'num_private',
                        'population', 'construction_year']

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
    # Lat/Long
    ##################################################
    # dodoma = (-6.173056, 35.741944)
    df['dodoma_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.distance((-6.173056, 35.741944), (x[0], x[1])).km, axis=1)


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

    # Convert np.nan and "none"/"nan" to special string "__NAN__"
    df['wpt_name'] = df['wpt_name'].replace("none", '__NAN__')
    df['public_meeting'] = df['public_meeting'].replace("nan", '__NAN__')
    df['permit'] = df['permit'].replace("nan", '__NAN__')
    df[cat_cols] = df[cat_cols].fillna('__NAN__')
    
    
    # Categorical levels "smushing" - convert long-tail values to "__OTHER__"
    for c in ['wpt_name', 'funder', 'extraction_type', 'installer', 'subvillage', 'lga', 'ward']:
        if train:
            top_n_values[c] = get_top_n_values(df, c, n=keep_top)
        df = keep_only(df, c, top_n_values[c])
        
    df[cat_cols] = df[cat_cols].astype('category')

    # Encoding
    #cat_cols = ['funder']
    if enc is not None:
        if train:
            enc.fit(df[cat_cols], y)
        _new_cols = enc.transform(df[cat_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_cols = pd.DataFrame(_new_cols, columns=["{}_{}".format('enc', i) for i in range(_new_cols.shape[1])])
        df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
        df = df.drop(cat_cols, axis=1)
    #else:
    #    df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)

    ##################################################
    # Dropping - don't need anymore
    ##################################################
    drop_cols = ['date_recorded']
    df = df.drop(drop_cols, axis=1)

    return df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder", help="Name of encoder to use? None, ordinal, mestimate, backward, hashing", default="None")

    parser.add_argument(
        "--keep-top", help="Number of levels in cat features to keep.", nargs='?', type=int, const=1, default=20)
    
    parser.add_argument(
        "--enc-dim", help="For hashing encoder, number of dimentions.", nargs='?', type=int, const=1, default=8)
    
    parser.add_argument(
        "--search-type", help="FLAML or randomCV?", default="flaml")
    
    parser.add_argument(
        "--search-time", help="FLAML time budget", default=1200)
        
        
    args = parser.parse_args()
    runname = str(uuid.uuid4())

    print("Run name: {}".format(runname))

    # This structure will hold all the results and will be dumped to disk.
    results = {}
    results['runname'] = runname
    results['starttime'] = str(datetime.datetime.now())
    results['hostname'] = socket.gethostname()
    results['encoder'] = args.encoder
    results['keep_top'] = args.keep_top
    results['enc_dim'] = args.enc_dim
    results['search_type'] = args.search_type
    results['search_time'] = args.search_time

    #df = pd.read_csv("https://drive.google.com/uc?export=download&id=1O3gYw1FlsbDYrXhma5_N6AYqQ3OKI3uh", parse_dates=['date_recorded'])
    dfo = pd.read_csv("data/pump_train.csv", parse_dates=['date_recorded'])

    Xo = dfo.drop('status_group', axis=1)
    y = dfo['status_group']


    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    
    enc = None
    if args.encoder == "ordinal":
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)
    elif args.encoder == "mestimate":
        enc = ce.wrapper.PolynomialWrapper(ce.m_estimate.MEstimateEncoder(randomized=True, verbose=2))
    elif args.encoder == "backward":
        enc = ce.backward_difference.BackwardDifferenceEncoder(handle_unknown='value', return_df=True)
    elif args.encoder == "hashing":
        enc = ce.hashing.HashingEncoder(return_df = True, n_components=args.enc_dim)
    elif args.encoder == "None":
        enc = None
    else:
        print("Error: undefined encoder: {}".format(args.encoder))
        exit()
        
    top_n_values = {}
    X = hack(Xo, y, imputer, top_n_values, enc, train=True, keep_top=args.keep_top)
    
    results['X_head'] = X.head().to_dict()
    
    #test_df = pd.read_csv('https://drive.google.com/uc?export=download&id=1Qnrd0pIRHJNoqNXNEDfp4YJglF4mRL_6', parse_dates=['date_recorded'])
    test_df = pd.read_csv('data/pump_test.csv', parse_dates=['date_recorded'])
    _test = hack(test_df, None, imputer, top_n_values, enc, train=False, keep_top=args.keep_top)   
    
    results['_test_head'] = _test.head().to_dict()
    
    #print(X[['funder', 'installer', 'wpt_name', 'basin', 'subvillage', 'region', 'lga', 'ward', 'public_meeting']].head(20))
    #print(X[['enc_0', 'enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5', 'enc_6', 'enc_7', 'enc_8', 'enc_9']].head(20))
    #print(X.head(20))
    #X.to_csv('out/X.csv', index=False)
    #_test.to_csv('out/_test.csv', index=False)
    #quit()
    
    pipe = None
    if args.search_type == "randomCV":
        
        from scipy.stats import uniform, randint

        params = {
         'n_jobs': 4,}
        clf = lgb.LGBMClassifier(**params)


        param_grid = {
            'colsample_bytree': uniform(0.6, 0.4),
            'max_depth': randint(5, 30),
            'max_bin': randint(8, 128),
            'num_leaves': randint(16, 512),
            'path_smooth': uniform(0.0, 0.2),
            'n_estimators': randint(25, 1000),
            'min_child_samples': randint(5, 100),
            'learning_rate': uniform(0.001, 0.9),
            'subsample': uniform(0.5, 0.5),
            'reg_alpha': uniform(0.0, 1.0),
            'reg_lambda': uniform(0.0, 1.0),
            'class_weight':['balanced', None],
        }

        pipe = RandomizedSearchCV(clf, param_grid, n_iter=1000, n_jobs=10, cv=3, scoring='accuracy', return_train_score=True, verbose=10)

        print("Running GridSearchCV")
        pipe.fit(X, y)

        results['cv_results_'] = pipe.cv_results_
        tbl = cv_results_to_df(pipe.cv_results_)
        tbl.to_csv("out/{}-cv_results.csv".format(runname), index=False)
                                  

    elif args.search_type == "flaml":
        
        pipe = AutoML()
        automl_settings = {
            "time_budget": args.search_time,
            "task": 'classififcation',
            "log_file_name": "out/flaml-{}.log".format(runname),
            "n_jobs": 10,
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
        
    if pipe is not None:
     

        preds = pipe.predict(_test)
        submission = pd.DataFrame(data={'id': test_df['id'], 'status_group': preds})
        submission.to_csv("out/{}-stepthom_submission.csv".format(runname), index=False)
        
    results['endtime'] = str(datetime.datetime.now())
    dump_results(runname, results)
    
    print("Run name: {}".format(runname))


if __name__ == "__main__":
    main()
