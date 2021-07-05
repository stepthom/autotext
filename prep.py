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


def add_indicator(X, col, missing_vals=[], replace_val=np.nan):
    col_new = "{}_missing".format(col)
    X[col_new] = X[col].apply(lambda x: 1 if x in missing_vals else 0)
    for missing_val in missing_vals:
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

# Only does the very basic; leaves more on the table for AutoML packages
# - Change types
# - Add np.nan indicators
def hack_simple(X, y=None):
    df = X.copy()
    
    ##################################################
    # Change Types
    ##################################################
    df['construction_year']= pd.to_numeric(df['construction_year'])
    df['public_meeting'] = df['public_meeting'].astype('str')
    df['permit'] = df['permit'].astype('str')

    ##################################################
    # Add missing value indicators
    ##################################################
    
    df = add_indicator(df, 'funder', ['0', np.nan])
    df = add_indicator(df, 'installer', ['0', np.nan])
    
    df = add_indicator(df, 'wpt_name', [np.nan, "none"])
    df = add_indicator(df, 'public_meeting', [np.nan, "nan"])
    df = add_indicator(df, 'permit', [np.nan, "nane"])
    
    # TODO: replace with nan (so it will be imputed later)?
    df = add_indicator(df, 'construction_year', [0], 1950)
    
    # TODO: replace with something else?
    df = add_indicator(df, 'amount_tsh', [0])

    df = add_indicator(df, 'population', [0])
    df = add_indicator(df, 'latitude', [-2e-08])
    df = add_indicator(df, 'longitude', [0])
    df = add_indicator(df, 'gps_height', [0])
    
    return df
    

def hack(X, y=None, imputer=None, top_n_values=None, enc=None, val_by_regions=None, train=False, keep_top=10):

    df = X.copy()
    
    ##################################################
    # Dropping - Won't need at all
    ##################################################
    # Note: need to leave ID, for downstream tasks
    drop_cols = ['recorded_by', 'num_private', 'scheme_name']
    dup_cols = ['payment_type', 'quantity_group', 'source_type', 'waterpoint_type_group']
    drop_cols = drop_cols + dup_cols
    df = df.drop(drop_cols, axis=1)

    ##################################################
    # Change Types
    ##################################################
    df['construction_year']= pd.to_numeric(df['construction_year'])
    df['public_meeting'] = df['public_meeting'].astype('str')
    df['permit'] = df['permit'].astype('str')

    ##################################################
    # Add missing value indicators
    ##################################################
    
    df = add_indicator(df, 'funder', ['0', np.nan])
    df = add_indicator(df, 'installer', ['0', np.nan])
    
    df = add_indicator(df, 'wpt_name', [np.nan, "none"])
    df = add_indicator(df, 'public_meeting', [np.nan, "nan"])
    df = add_indicator(df, 'permit', [np.nan, "nane"])
    
    # TODO: replace with nan (so it will be imputed later)?
    df = add_indicator(df, 'construction_year', [0], 1950)
    
    # TODO: replace with something else?
    df = add_indicator(df, 'amount_tsh', [0])

    df = add_indicator(df, 'population', [0])
    df = add_indicator(df, 'latitude', [-2e-08])
    df = add_indicator(df, 'longitude', [0])
    df = add_indicator(df, 'gps_height', [0])
    
    
    ####################################################
    # Special Impute: replace with region means
    ####################################################
    print("DEBUG: hack: special imputing")
    # Note: the following only works since we've added indicators and replaced "missing" values with np.nan.

    simpute_cols = ['gps_height', 'population', 'latitude', 'longitude']
    
    if train:
        # During training, need to compute the mean value of this column for each region, save for later
        for c in simpute_cols:
            mean_val = df[c].mean(skipna=True)
            val_by_regions[c] = df.groupby('region_code').agg({c: lambda x: x.mean(skipna=True)}).reset_index().fillna(mean_val)

    # Helper function to replace mean from region if missing
    def f(row, col, val_by_region):
        if row["{}_missing".format(col)] == 1:
            return val_by_region[val_by_region['region_code'] == row['region_code']][col].item()
        else:
            return row[col]
    
    for c in simpute_cols:
        df[c] = df.apply(lambda row: f(row, c, val_by_regions[c]), axis=1)
    

    
    ##################################################
    # Impute missing numeric values
    ##################################################
    print("DEBUG: hack: imputing")
    # Impute Missing Value
    numeric_features = ['amount_tsh', 
                        #'gps_height', 
                        #'longitude', 
                        #'latitude', 
                        #'population', 
                        'construction_year']

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
            
    print("DEBUG: hack: cat_cols = {}".format(cat_cols))
            
    # Convert np.nan and "none"/"nan" to special string "__NAN__"
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
        "--encoder", help="Name of encoder to use? None, ordinal, mestimate, backward, glm, hashing, codes", default="None")

    parser.add_argument(
        "--keep-top", help="Number of levels in cat features to keep.", nargs='?', type=int, const=1, default=20)
    
    parser.add_argument(
        "--enc-dim", help="For hashing encoder, number of dimentions.", nargs='?', type=int, const=1, default=8)
    
    parser.add_argument(
        "--simple", help="Only do simple preprocessing.", default=False, action='store_true')
    
    args = parser.parse_args()


    #dfo = pd.read_csv("https://drive.google.com/uc?export=download&id=1O3gYw1FlsbDYrXhma5_N6AYqQ3OKI3uh", parse_dates=['date_recorded'])
    #test_df = pd.read_csv('https://drive.google.com/uc?export=download&id=1Qnrd0pIRHJNoqNXNEDfp4YJglF4mRL_6', parse_dates=['date_recorded'])
    
    train_df= pd.read_csv(args.train_input, parse_dates=['date_recorded'])
    test_df = pd.read_csv(args.test_input, parse_dates=['date_recorded'])
    
    target = 'status_group'

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
    val_by_regions = {}
    
    X_train = None
    X_test = None
    fn_id = None
    if args.simple:
        X_train = hack_simple(train_df.drop(target, axis=1), train_df[target])
        X_test = hack_simple(test_df, None)
        fn_id = "simple"
    else:
        X_train = hack(train_df.drop(target, axis=1), train_df[target], imputer, top_n_values, enc, val_by_regions, train=True, keep_top=args.keep_top)
        X_test = hack(test_df, None, imputer, top_n_values, enc, val_by_regions, train=False, keep_top=args.keep_top)
        fn_id = "{}_{}_{}".format(args.encoder, args.keep_top, args.enc_dim)
    
    X_train[target] = train_df[target]
    
    def append_string(filename, fn_id):
        name, ext = os.path.splitext(filename)
        return "{name}_{fn_id}{ext}".format(name=name, fn_id=fn_id, ext=ext)
   
    fn_train = append_string(args.train_input, fn_id)
    X_train.to_csv(fn_train, index=False)
    print("DEBUG: Wrote file {}".format(fn_train))
    
    fn_test = append_string(args.test_input, fn_id)
    X_test.to_csv(fn_test, index=False)
    print("DEBUG: Wrote file {}".format(fn_test))

if __name__ == "__main__":
    main()
