import argparse
import numpy as np
import uuid

import os
import jsonpickle
import json
import datetime
import socket

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import KernelPCA, PCA, TruncatedSVD
from sklearn.impute import MissingIndicator
from pandas.api.types import is_numeric_dtype

import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper

import geopy.distance

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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


def hack(X, y=None, train=False, 
         target_col="status_group",
         id_col="id", 
         num_imputer=None, 
         num_indicator=None, 
         top_n_values=None, 
         enc=None, 
         val_by_regions=None, 
         special_impute_cols=[],
         keep_top=10):

    df = X.copy()
    
    msgs=[]
    
    ##################################################
    # Dropping - Won't need at all
    ##################################################
    # Note: need to leave ID, for downstream tasks
    drop_cols = ['recorded_by', 'num_private', 'scheme_name']
    dup_cols = ['payment_type', 'quantity_group', 'source_type', 'waterpoint_type_group']
    drop_cols = drop_cols + dup_cols
    msgs.append('Dropping cols: {}'.format(drop_cols))
    df = df.drop(drop_cols, axis=1)
    
    ##################################################
    # Gather data types
    ##################################################
    
    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c == id_col:
            continue
        col_type = df[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            cat_cols.append(c)
        elif is_numeric_dtype(df[c]):
            num_cols.append(c)
            
    #num_cols = df.select_dtypes(include="number")
            
    msgs.append('Cat cols: {}'.format(cat_cols))
    msgs.append('Num cols: {}'.format(num_cols))
    print("DEBUG: Cat cols: {}".format(cat_cols))
    print("DEBUG: Num cols: {}".format(num_cols))
    

    ##################################################
    # Change Types
    ##################################################
    msgs.append('Changing construction_year to numeric')
    df['construction_year']= pd.to_numeric(df['construction_year'])
    
    msgs.append('Changing public_meeting to str')
    df['public_meeting'] = df['public_meeting'].astype('str')
    
    msgs.append('Changing permit to str')
    df['permit'] = df['permit'].astype('str')
    

    ##################################################
    # Weird values to np.nan
    ##################################################
    
    strzero_cols = ['funder', 'installer']
    msgs.append("strzero_cols: {}".format(strzero_cols))
    for col in strzero_cols:
        df[col] = df[col].replace('0', np.nan)
        
    none_cols = ['wpt_name']
    msgs.append("none_cols: {}".format(none_cols))
    for col in none_cols:
        df[col] = df[col].replace('none', np.nan)
        
    nan_cols = ['public_meeting', 'permit']
    msgs.append("nan_cols: {}".format(nan_cols))
    for col in nan_cols:
        df[col] = df[col].replace('nan', np.nan)
        
    zero_cols = ['amount_tsh', 'population', 'longitude', 'gps_height']
    msgs.append("zero_cols: {}".format(zero_cols))
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
            
    msgs.append("Round latitude to 5 places") 
    df['latitude'] = df['latitude'].round(decimals = 5)
        
    zero_cols = ['amount_tsh', 'population', 'longitude', 'latitude', 'gps_height']
    msgs.append("zero_cols: {}".format(zero_cols))
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
                
    ##################################################
    # Numeric: Add missing value indicators
    ##################################################
    
    # Must happen before numbers are imputed!
    if num_indicator is not None:
        print("DEBUG: hack: missing value indicator for numeric")
        if train:
            num_indicator = num_indicator.fit(df[num_cols], y)
    
        _new_cols = num_indicator.transform(df[num_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_cols = pd.DataFrame(_new_cols, columns=["missing_{}".format(c) for c in num_cols])
        df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
        
    
    ####################################################
    # Numeric: Special Impute: replace with region means
    ####################################################
    print("DEBUG: hack: special imputing")
    # Note: the following only works since we've added indicators and replaced "missing" values with np.nan.

    msgs.append("special_impute_cols: {}".format(special_impute_cols))
                
    if train:
        # During training, need to compute the mean value of this column for each region, save for later
        for c in special_impute_cols:
            mean_val = df[c].mean(skipna=True)
            val_by_regions[c] = df.groupby('region_code').agg({c: lambda x: x.mean(skipna=True)}).reset_index().fillna(mean_val)

    # Helper function to replace mean from region if missing
    def f(row, col, val_by_region):
        if np.isnan(row[col]):
            return val_by_region[val_by_region['region_code'] == row['region_code']][col].item()
        else:
            return row[col]
        
    
    for c in special_impute_cols:
        df[c] = df.apply(lambda row: f(row, c, val_by_regions[c]), axis=1)
    
    
    ##################################################
    # Numeric: Impute missing  values
    ##################################################
    print("DEBUG: hack: simple numeric imputing")
    # Impute Missing Value
    simple_impute_cols = list(set(num_cols) - set(special_impute_cols)) 
                
    msgs.append('Numeric simple imputing for cols: {}'.format(simple_impute_cols))

    if train:
        num_imputer = num_imputer.fit(df[simple_impute_cols], y)
    df[simple_impute_cols] = num_imputer.transform(df[simple_impute_cols])

   
    ##################################################
    # Date/Time
    ##################################################
    print("DEBUG: hack: date/time")
                
    msgs.append('Adding date/time features for date_recorded')
    df = add_datetime(df, 'date_recorded')

    baseline = pd.datetime(2014, 1, 1)
    df['date_recorded_since'] = (baseline - df['date_recorded']).dt.days

    df['timediff'] = df['year_date_recorded'] - df['construction_year']
    
    ##################################################
    # Lat/Long
    ##################################################
    print("DEBUG: hack: lat long")
    
    msgs.append('Adding distances for four cities')
    daressalaam = (-6.8, 39.283333)
    mwanza = (-2.516667, 32.9)
    arusha = (-3.366667, 36.683333)
    dodoma = (-6.173056, 35.741944)
    df['daressallam_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(daressalaam, (x[0], x[1])).km, axis=1)
    df['mwanza_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(mwanza, (x[0], x[1])).km, axis=1)
    df['arusha_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(arusha, (x[0], x[1])).km, axis=1)
    df['dodoma_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(dodoma, (x[0], x[1])).km, axis=1)

    
    ##################################################
    # Categorical Imputing
    ##################################################
                
    # Impute categorical? 
    # - If yes, use top value count
    # - If no, replace nan with special string "__NAN__"
    print("DEBUG: hack: imputing categorical")
                
    # TODO: not implemented yet
    df[cat_cols] = df[cat_cols].fillna('__NAN__')
    df[cat_cols] = df[cat_cols].astype('category')

    ##################################################
    # Categorical Smushing
    ##################################################
      
    # Categorical levels "smushing" - convert long-tail values to "__OTHER__"
    df[cat_cols] = df[cat_cols].astype(str)
    for c in cat_cols:
        if train:
            top_n_values[c] = get_top_n_values(df, c, n=keep_top)
        df = keep_only(df, c, top_n_values[c])
        
    df[cat_cols] = df[cat_cols].astype('category')
                
    ##################################################
    # Categorical Encoding
    ##################################################

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
    msgs.append("Dropping cols: {}".format(drop_cols))
    df = df.drop(drop_cols, axis=1)

    return df, msgs

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        print("DEBUG: obj: {}".format(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dump_results(runname, results):

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__
        
    with open('data/{}-log.json'.format(runname), 'w') as fp:
        json.dump(results, fp, indent=4, cls=NumpyEncoder)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train-input", help="train file name", default="data/pump_train.csv")
    parser.add_argument("-s", "--test-input", help="test file name", default="data/pump_test.csv")
    args = parser.parse_args()
    
    train_df= pd.read_csv(args.train_input, parse_dates=['date_recorded'])
    test_df = pd.read_csv(args.test_input, parse_dates=['date_recorded'])
    
    target_col = "status_group"
    id_col = "id" 
    
    num_indicators = [
       None,
       MissingIndicator(features="all")
    ]
    
    num_imputers = [
       SimpleImputer(missing_values=np.nan, strategy="median") 
    ]

    
    cat_encoders = [
        None,
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32),
        ce.wrapper.PolynomialWrapper(ce.m_estimate.MEstimateEncoder(randomized=False, verbose=0)),
        #ce.hashing.HashingEncoder(return_df = True, n_components=args.enc_dim),
    ]
    
    keep_tops = [10, 20, 50, 75, 100, 200]
    
    special_impute_colss = [
        [],
        ['gps_height', 'population', 'latitude', 'longitude']
    ]
    
    for num_indicator in num_indicators:
        for num_imputer in num_imputers:
            for cat_encoder in cat_encoders:
                for keep_top in keep_tops:
                    for special_impute_cols in special_impute_colss:
                        
                            results = {}
                            data_id = str(uuid.uuid4())
                            results['data_id'] = data_id
                            results['starttime'] = str(datetime.datetime.now())
                            results['hostname'] = socket.gethostname()
                            results['args'] = vars(args)
                        
                            results['num_indicator'] = str(num_indicator)
                            results['num_imputer'] = str(num_imputer)
                            results['cat_encoder'] = str(cat_encoder)
                            results['keep_top'] = keep_top
                            results['special_impute_cols'] = special_impute_cols
                            
                            config_summary = "{}_{}_{}_{}_{}".format(str(num_indicator), 
                                                                     str(num_imputer),
                                                                     str(cat_encoder), 
                                                                     keep_top, 
                                                                     special_impute_cols)
                            
                            results['config_summary'] = config_summary
                            
                            top_n_values = {}
                            val_by_regions = {}

                            hack_args = {
                                "target_col": target_col,
                                "id_col": id_col,
                                "num_imputer": num_imputer,
                                "num_indicator": num_indicator,
                                "enc": cat_encoder,
                                "top_n_values": top_n_values,
                                "val_by_regions": val_by_regions,
                                "special_impute_cols": special_impute_cols,
                                "keep_top": keep_top,
                            } 
    
                            X_train, train_msgs = hack(train_df.drop(target_col, axis=1), train_df[target_col], train=True, **hack_args)
                            X_train[target_col] = train_df[target_col]
                            results['train_msgs'] = train_msgs

                            X_test, test_msgs = hack(test_df, None, train=False, **hack_args)
                            results['test_msgs'] = test_msgs

                            results['top_n_values'] = top_n_values
                            results['val_by_regions'] = {}
                            for col in val_by_regions:
                                results['val_by_regions'][col] = val_by_regions[col].to_dict()

                            def append_string(filename, data_id):
                                name, ext = os.path.splitext(filename)
                                return "{name}_{fn_id}{ext}".format(name=name, fn_id=data_id, ext=ext)

                            fn_train = append_string(args.train_input, data_id)
                            X_train.to_csv(fn_train, index=False)
                            print("DEBUG: Wrote file {}".format(fn_train))

                            fn_test = append_string(args.test_input, data_id)
                            X_test.to_csv(fn_test, index=False)
                            print("DEBUG: Wrote file {}".format(fn_test))

                            results['fn_train'] = fn_train
                            results['fn_test'] = fn_test

                            results['endtime'] = str(datetime.datetime.now())
                            dump_results(data_id, results)
                            print("config_summary: {}".format(config_summary))
                            print("Data ID: {}".format(data_id))

if __name__ == "__main__":
    main()
