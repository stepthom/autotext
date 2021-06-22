import argparse
import numpy as np

import os

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from datetime import date, datetime
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD

import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper

import geopy.distance

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
    # Note: need to leave ID, for downstream tasks
    drop_cols = ['num_private']
    dup_cols = [] # ['payment_type', 'quantity_group', 'source_type', 'waterpoint_type_group']
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
    df = add_indicator(df, 'construction_year', 0, 1950)
    
    # TODO: replace with something else?
    df = add_indicator(df, 'amount_tsh', 0)

    # TODO: replace with region means?
    df = add_indicator(df, 'population', 0)
    df = add_indicator(df, 'latitude', -2e-08)
    df = add_indicator(df, 'longitude', 0)
    df = add_indicator(df, 'gps_height', 0)

    ##################################################
    # Impute missing numeric values
    ##################################################
    print("DEBUG: hack: imputing")
    # Impute Missing Value
    numeric_features = ['amount_tsh', 'gps_height', 
                        'longitude', 'latitude', 
                        'population', 'construction_year']

    if train:
        imputer = imputer.fit(df[numeric_features], y)
    df[numeric_features] = imputer.transform(df[numeric_features])

   
    ##################################################
    # Date/Time
    ##################################################
    print("DEBUG: hack: date/time")
    df = add_datetime(df, 'date_recorded')

    baseline = pd.datetime(2014, 1, 1)
    df['date_recorded_since'] = (baseline - df['date_recorded']).dt.days

    df['timediff'] = df['year_date_recorded'] - df['construction_year']
    
    ##################################################
    # Lat/Long
    ##################################################
    print("DEBUG: hack: lat long")
    daressalaam = (-6.8, 39.283333)
    mwanza = (-2.516667, 32.9)
    arusha = (-3.366667, 36.683333)
    dodoma = (-6.173056, 35.741944)
    df['daressallam_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(daressalaam, (x[0], x[1])).km, axis=1)
    df['mwanza_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(mwanza, (x[0], x[1])).km, axis=1)
    df['arusha_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(arusha, (x[0], x[1])).km, axis=1)
    df['dodoma_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(dodoma, (x[0], x[1])).km, axis=1)


    ##################################################
    # Categorical
    ##################################################
    print("DEBUG: hack: categorical")
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
    if enc is None:
        print("DEBUG: hack: encoding: raw")
        # Do nothing - leave them "raw"
        pass
    elif isinstance(enc, str) and enc=="codes":
        print("DEBUG: hack: encoding: codes")
        df[cat_cols] = df[cat_cols].apply(lambda x: x.cat.codes)
    else:
        print("DEBUG: hack: encoding: encoder")
        if train:
            enc.fit(df[cat_cols], y)
        _new_cols = enc.transform(df[cat_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_cols = pd.DataFrame(_new_cols, columns=["{}_{}".format('enc', i) for i in range(_new_cols.shape[1])])
        df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
        df = df.drop(cat_cols, axis=1)
        

    ##################################################
    # Dropping - don't need anymore
    ##################################################
    drop_cols = ['date_recorded']
    df = df.drop(drop_cols, axis=1)

    return df


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--train-input", help="train file name", default="data/pump_train.csv")
    
    parser.add_argument(
        "--test-input", help="test file name", default="data/pump_test.csv")
    
    parser.add_argument(
        "--output_str", help="String to insert into output file name", default="data/pump_test.csv")
    
    parser.add_argument(
        "--encoder", help="Name of encoder to use? None, ordinal, mestimate, backward, glm, hashing, codes", default="None")

    parser.add_argument(
        "--keep-top", help="Number of levels in cat features to keep.", nargs='?', type=int, const=1, default=20)
    
    parser.add_argument(
        "--enc-dim", help="For hashing encoder, number of dimentions.", nargs='?', type=int, const=1, default=8)
    
    args = parser.parse_args()


    #dfo = pd.read_csv("https://drive.google.com/uc?export=download&id=1O3gYw1FlsbDYrXhma5_N6AYqQ3OKI3uh", parse_dates=['date_recorded'])
    #test_df = pd.read_csv('https://drive.google.com/uc?export=download&id=1Qnrd0pIRHJNoqNXNEDfp4YJglF4mRL_6', parse_dates=['date_recorded'])
    
    dfo     = pd.read_csv(args.train_input, parse_dates=['date_recorded'])
    test_df = pd.read_csv(args.test_input, parse_dates=['date_recorded'])
    
    target = 'status_group'

    Xo = dfo.drop(target, axis=1)

    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    
    enc = None
    if args.encoder == "ordinal":
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)
    elif args.encoder == "mestimate":
        enc = ce.wrapper.PolynomialWrapper(ce.m_estimate.MEstimateEncoder(randomized=False, verbose=0))
    elif args.encoder == "glm":
        enc = ce.wrapper.PolynomialWrapper(ce.glmm.GLMMEncoder(return_df=True))
    elif args.encoder == "backward":
        enc = ce.backward_difference.BackwardDifferenceEncoder(handle_unknown='value', return_df=True)
    elif args.encoder == "hashing":
        enc = ce.hashing.HashingEncoder(return_df = True, n_components=args.enc_dim)
    elif args.encoder == "None":
        enc = None
    elif args.encoder == "codes":
        enc = "codes"
    else:
        print("Error: undefined encoder: {}".format(args.encoder))
        exit()
        
    top_n_values = {}
    
    X = hack(dfo.drop(target, axis=1), dfo[target], imputer, top_n_values, enc, train=True, keep_top=args.keep_top)
    X[target] = dfo[target]
    
    _test = hack(test_df, None, imputer, top_n_values, enc, train=False, keep_top=args.keep_top)   
    
    id = "{}_{}_{}".format(args.encoder, args.keep_top, args.enc_dim)
    
    def append_string(filename, id):
        name, ext = os.path.splitext(filename)
        return "{name}_{id}{ext}".format(name=name, id=id, ext=ext)
   
    _X_fn = append_string(args.train_input, id)
    X.to_csv(_X_fn, index=False)
    
    _test_fn = append_string(args.test_input, id)
    _test.to_csv(_test_fn, index=False)
    
    print("Wrote files {} and {}.".format(_X_fn, _test_fn))


if __name__ == "__main__":
    main()
