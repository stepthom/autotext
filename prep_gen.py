import argparse
import numpy as np
import os
import jsonpickle
import json
import datetime
import socket
import pandas as pd
import uuid

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
from pandas.api.types import is_numeric_dtype

import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper

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


def hack(X, y=None, train=False, 
         target_col="status_group",
         id_col="id", 
         num_imputer=None, 
         num_indicator=None, 
         top_n_values=None, 
         enc=None, 
         keep_top=10,
         impute_cat=False):

    df = X.copy()
    
    msgs=[]
    
    
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
            
    msgs.append('Cat cols: {}'.format(cat_cols))
    msgs.append('Num cols: {}'.format(num_cols))
    print("DEBUG: Cat cols: {}".format(cat_cols))
    print("DEBUG: Num cols: {}".format(num_cols))
    

    ##################################################
    # Change Types
    ##################################################
    #msgs.append('Changing construction_year to numeric')
    #df['construction_year']= pd.to_numeric(df['construction_year'])
    
    #msgs.append('Changing public_meeting to str')
    #df['public_meeting'] = df['public_meeting'].astype('str')
    

    ##################################################
    # Weird values to np.nan
    ##################################################
    

                
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
    
    
    ##################################################
    # Numeric: Impute missing  values
    ##################################################
    print("DEBUG: hack: simple numeric imputing")
    # Impute Missing Value
    simple_impute_cols = num_cols
                
    msgs.append('Numeric simple imputing for cols: {}'.format(simple_impute_cols))

    if train:
        num_imputer = num_imputer.fit(df[simple_impute_cols], y)
    df[simple_impute_cols] = num_imputer.transform(df[simple_impute_cols])

    
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
    drop_cols = []
    msgs.append("Dropping cols: {}".format(drop_cols))
    #df = df.drop(drop_cols, axis=1)

    return df, msgs

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        print("DEBUG: obj: {}".format(obj))
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dump_json(fn, json_obj):

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__
        
    print("DEBUG: Writing json file {}".format(fn))    
    with open(fn, 'w') as fp:
        json.dump(json_obj, fp, indent=4, cls=NumpyEncoder)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train-input", help="train file name", default="h1n1/vaccine_h1n1_train.csv")
    parser.add_argument("-s", "--test-input", help="test file name", default="h1n1/vaccine_h1n1_test.csv")
    parser.add_argument("-d", "--data-dir", help="output data dir", default="h1n1/data")
    parser.add_argument("-a", "--target-col", help="name of target col", default="h1n1_vaccine")
    parser.add_argument("-i", "--id-col", help="name of id col", default="respondent_id")
    args = parser.parse_args()
    
    train_df= pd.read_csv(args.train_input)
    test_df = pd.read_csv(args.test_input)
    
    data_dir = args.data_dir
    target_col = args.target_col
    id_col = args.id_col
    
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
        ce.hashing.HashingEncoder(return_df = True, n_components=8),
        ce.hashing.HashingEncoder(return_df = True, n_components=32),
    ]
    
    keep_tops = [25]
    
    impute_cats = [False]
    
    for num_indicator in num_indicators:
        for num_imputer in num_imputers:
            for cat_encoder in cat_encoders:
                for keep_top in keep_tops:
                    for impute_cat in impute_cats:
                        
                        data_sheet = {}
                        data_id = str(uuid.uuid4())
                        data_sheet['data_id'] = data_id
                        data_sheet['starttime'] = str(datetime.datetime.now())
                        data_sheet['hostname'] = socket.gethostname()
                        data_sheet['args'] = vars(args)

                        data_sheet['num_indicator'] = str(num_indicator)
                        data_sheet['num_imputer'] = str(num_imputer)
                        data_sheet['cat_encoder'] = str(cat_encoder)
                        data_sheet['keep_top'] = keep_top
                        data_sheet['impute_cat'] = impute_cat

                        config_summary = "{}_{}_{}_{}_{}".format(str(num_indicator), 
                                                                 str(num_imputer),
                                                                 str(cat_encoder), 
                                                                 keep_top, 
                                                                 impute_cat)

                        data_sheet['config_summary'] = config_summary

                        top_n_values = {}
                        val_by_regions = {}

                        hack_args = {
                            "target_col": target_col,
                            "id_col": id_col,
                            "num_imputer": num_imputer,
                            "num_indicator": num_indicator,
                            "enc": cat_encoder,
                            "top_n_values": top_n_values,
                            "impute_cat": impute_cat,
                            "keep_top": keep_top,
                        } 

                        X_train, train_msgs = hack(train_df.drop(target_col, axis=1), train_df[target_col], train=True, **hack_args)
                        X_train[target_col] = train_df[target_col]
                        data_sheet['train_msgs'] = train_msgs

                        X_test, test_msgs = hack(test_df, None, train=False, **hack_args)
                        data_sheet['test_msgs'] = test_msgs

                        data_sheet['top_n_values'] = top_n_values

                        
                        def get_fn(filename, data_dir, data_id):
                            # filename will be somethingl like: h1n1/test.csv
                                                    # output will be data_dir/test_data_id.csv
                            name, ext = os.path.splitext(os.path.basename(filename))
                            return os.path.join(data_dir, "{}_{}{}".format(name, data_id, ext))

                        fn_train = get_fn(args.train_input, data_dir, data_id)
                        X_train.to_csv(fn_train, index=False)
                        print("DEBUG: Wrote file {}".format(fn_train))

                        fn_test = get_fn(args.test_input, data_dir, data_id)
                        X_test.to_csv(fn_test, index=False)
                        print("DEBUG: Wrote file {}".format(fn_test))

                        data_sheet['fn_train'] = fn_train
                        data_sheet['fn_test'] = fn_test

                        data_sheet['endtime'] = str(datetime.datetime.now())
                        
                        data_sheet_fn = os.path.join(data_dir, "{}.json".format(data_id))
                        dump_json(data_sheet_fn, data_sheet)
                        print("DEBUG: config_summary: {}".format(config_summary))
                        print("DEBUG: Data ID: {}".format(data_id))

if __name__ == "__main__":
    main()
