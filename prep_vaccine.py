import argparse
import numpy as np

import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator

import category_encoders as ce

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


def hack(X, y=None, num_imputer=None, num_indicator=None, enc=None, train=False):

    df = X.copy()
    
    cat_cols = []
    num_cols = []
    for c in df.columns:
        if c == "respondent_id":
            continue
        col_type = df[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            cat_cols.append(c)
        else:
            num_cols.append(c)
            
    print("DEBUG: hack: cat_cols = {}".format(cat_cols))
    print("DEBUG: hack: num_cols = {}".format(num_cols))

    
    ##################################################
    # Dropping - Won't need at all
    ##################################################
    # Note: need to leave ID, for downstream tasks
    #drop_cols = ['recorded_by', 'num_private', 'scheme_name']
    #dup_cols = ['payment_type', 'quantity_group', 'source_type', 'waterpoint_type_group']
    #drop_cols = drop_cols + dup_cols
    #df = df.drop(drop_cols, axis=1)

    ##################################################
    # Change Types
    ##################################################
    #df['construction_year']= pd.to_numeric(df['construction_year'])
    #df['public_meeting'] = df['public_meeting'].astype('str')
    #df['permit'] = df['permit'].astype('str')

    ##################################################
    # Add missing value indicators
    ##################################################
    
    # Impute categorical? 
    # - If yes, use top value count
    # - If no, replace nan with special string "__NAN__"
    print("DEBUG: hack: imputing categorical")
    
    df[cat_cols] = df[cat_cols].fillna('__NAN__')
    df[cat_cols] = df[cat_cols].astype('category')
        
    
    # Impute numeric?
    #  - If yes, use median
    print("DEBUG: hack: imputing numeric")
    if train:
        num_imputer = num_imputer.fit(df[num_cols], y)
        if num_indicator is not None:
            num_indicator = num_indicator.fit(df[num_cols], y)
    
    # Must happen before numbers are imputed!
    if num_indicator is not None:
        _new_cols = num_indicator.transform(df[num_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_cols = pd.DataFrame(_new_cols, columns=["{}_{}".format('missing', c) for c in num_cols])
        df = pd.concat([df, _new_cols], axis=1, ignore_index=False)

            
    df[num_cols] = num_imputer.transform(df[num_cols])
    
  

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
    #drop_cols = ['date_recorded']
    #df = df.drop(drop_cols, axis=1)

    return df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train-input", help="train file name", default="data/pump_train.csv")
    parser.add_argument("-s", "--test-input", help="test file name", default="data/pump_test.csv")
    parser.add_argument("--target", help="h1n1_vaccine or seasonal_vaccine", default="h1n1_vaccine")
    parser.add_argument("-e", "--encoder", help="Name of encoder to use? None, ordinal, mestimate, backward, glm, hashing, codes", default="None")
    parser.add_argument("-m", '--missing-indicator', help="include missing indicator for numeric features?", default=False, action='store_true')
    parser.add_argument("-i", '--impute-cats', default=False, action='store_true')
    args = parser.parse_args()
    
    train_df= pd.read_csv(args.train_input)
    test_df = pd.read_csv(args.test_input)
    
    target = args.target

    num_imputer = SimpleImputer(missing_values=np.nan, strategy="median", add_indicator=False)
    
    num_indicator = None
    if args.missing_indicator:
        num_indicator = MissingIndicator(features="all")
    
    enc = None
    if args.encoder == "ordinal":
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32)
    elif args.encoder == "mestimate":
        enc = ce.m_estimate.MEstimateEncoder(randomized=False, verbose=0)
    elif args.encoder == "glm":
        enc = ce.glmm.GLMMEncoder(return_df=True)
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
    
    print(train_df.head())
    
    X_train = hack(train_df.drop(target, axis=1), train_df[target], num_imputer, num_indicator, enc, train=True)
    X_test = hack(test_df, None, num_imputer, num_indicator, enc, train=False)
    fn_id = "{}_{}_{}".format(args.encoder, args.missing_indicator, args.impute_cats)
    
    X_train[target] = train_df[target]
    
    def append_string(filename, fn_id):
        name, ext = os.path.splitext(filename)
        return "{name}_{fn_id}{ext}".format(name=name, fn_id=fn_id, ext=ext)
   
    fn_train = append_string(args.train_input, fn_id)
    print(X_train.head())
    X_train.to_csv(fn_train, index=False)
    print("DEBUG: Wrote file {}".format(fn_train))
    
    fn_test = append_string(args.test_input, fn_id)
    X_test.to_csv(fn_test, index=False)
    print("DEBUG: Wrote file {}".format(fn_test))

if __name__ == "__main__":
    main()
