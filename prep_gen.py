import argparse
import numpy as np
import os
import jsonpickle
import json
import datetime
import socket
import pandas as pd
import uuid
from autofeat import AutoFeatRegressor, AutoFeatClassifier, AutoFeatLight

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

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
         impute_cat=False,
         autofeat=None,
         dimreduc=None,
         feature_selector=None,
        ):

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
    
    all_cols = cat_cols + num_cols
    

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
        _new_cols = None
        if train:
            _new_cols = enc.fit_transform(df[cat_cols], y)
        else:
            _new_cols = enc.transform(df[cat_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_col_names =  ["{}_{}".format('enc_', i) for i in range(_new_cols.shape[1])]
            _new_cols = pd.DataFrame(_new_cols, columns=_new_col_names)
        df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
        df = df.drop(cat_cols, axis=1)
        
        
    ##################################################
    # Autofeat
    ##################################################
    
    if autofeat is not None:
        print("DEBUG: hack: Autofeats")
        msgs.append("Autofeats being called on {} features".format(len(num_cols)))
        _new_cols = None
        if train:
            _new_cols = autofeat.fit_transform(df[num_cols])
        else:
            _new_cols = autofeat.transform(df[num_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_col_names =  ["{}_{}".format('autofit', i) for i in range(_new_cols.shape[1])]
            _new_cols = pd.DataFrame(_new_cols, columns=_new_col_names)
        df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
        
        
    ##################################################
    # Dim reduction
    ##################################################
    
    if dimreduc is not None:
        dimreduc_cols = [col for col in list(df.columns) if c != id_col]
        print("DEBUG: hack: dimensionality reduction on {} cols".format(len(dimreduc_cols)))
        msgs.append("Dimreduc being called")
        _new_cols = None
        if train:
            _new_cols = dimreduc.fit_transform(df[dimreduc_cols])
        else:
            _new_cols = dimreduc.transform(df[dimreduc_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_col_names =  ["{}_{}".format('dimreduc', i) for i in range(_new_cols.shape[1])]
            _new_cols = pd.DataFrame(_new_cols, columns=_new_col_names)
        df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
        
    ##################################################
    # Feature selector
    ##################################################
    
    if feature_selector is not None:
        fs_cols = [col for col in list(df.columns) if c != id_col]
        print("DEBUG: hack: feature_selection on {} cols".format(len(fs_cols)))
        msgs.append("Feature selection being called")
        _new_cols = None
        if train:
            _new_cols = feature_selector.fit_transform(df[fs_cols], y)
        else:
            _new_cols = feature_selector.transform(df[fs_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_col_names =  ["{}_{}".format('feature_selector', i) for i in range(_new_cols.shape[1])]
            _new_cols = pd.DataFrame(_new_cols, columns=_new_col_names)
        df = _new_cols
        

    ##################################################
    # Dropping - don't need anymore
    ##################################################
    drop_cols = []
    msgs.append("Dropping cols: {}".format(drop_cols))
    #df = df.drop(drop_cols, axis=1)
    
    print("DEBUG: hack returning df shape {}".format(df.shape))

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
    parser.add_argument('-c', '--competition', default='seasonal')
    args = parser.parse_args()
   
    train_input = None
    test_input = None
    data_dir = None
    target_col = None
    id_col = None
    if args.competition.startswith("p"):
        train_input = "pump/.csv"
        test_input = "pump/.csv"
        data_dir = "pump/data"
        target_col = "status_group"
        id_col = "id"
        out_dir = "pump/out"
    elif args.competition.startswith("h"):
        train_input = "h1n1/vaccine_h1n1_train.csv"
        test_input = "h1n1/vaccine_h1n1_test.csv"
        data_dir = "h1n1/data"
        target_col = "h1n1_vaccine"
        id_col = "respondent_id"
    elif args.competition.startswith("s"):
        train_input = "seasonal/vaccine_seasonal_train.csv"
        test_input = "seasonal/vaccine_seasonal_test.csv"
        data_dir = "seasonal/data"
        target_col = "seasonal_vaccine"
        id_col = "respondent_id"
    elif args.competition.startswith("e"):
        train_input = "earthquake/earthquake_train.csv"
        test_input = "earthquake/earthquiake_test.csv"
        data_dir = "earthquake/data"
        target_col = "damage_grade"
        id_col = "building_id"
        out_dir = "earthquake/out"
    else:
        print("Error: unknown competition type: {}".format(args.competition))
        return
    
    
    train_df= pd.read_csv(train_input)
    test_df = pd.read_csv(test_input)
    
    num_indicators = [
       MissingIndicator(features="all"),
       None,
    ]
    
    num_imputers = [
       SimpleImputer(missing_values=np.nan, strategy="median"),
    ]

    cat_encoders = [
        ce.wrapper.PolynomialWrapper(ce.cat_boost.CatBoostEncoder(handle_unknown="value", sigma=None)),
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32),
        ce.wrapper.PolynomialWrapper(ce.m_estimate.MEstimateEncoder(randomized=False, verbose=0)),
    ]
    
    keep_tops = [25]
    
    impute_cats = [False]
    
    autofeats = [
        None,
        AutoFeatLight(verbose=1, compute_ratio=False, compute_product=True, scale=False),
    ]
                                              
    dimreducs = [
        KernelPCA(n_components=25, kernel='sigmoid', n_jobs=10, eigen_solver='arpack', max_iter=200),
        KernelPCA(n_components=25, kernel='poly', n_jobs=10, eigen_solver="arpack", max_iter=200),
        KernelPCA(n_components=25, kernel='rbf', n_jobs=10, eigen_solver="arpack", max_iter=200),
        None,
    ]
    
    feature_selectors = [
        None,
        SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=100, random_state=42), threshold=-np.inf, max_features=25),
    ]
    
    for num_indicator in num_indicators:
        for num_imputer in num_imputers:
            for cat_encoder in cat_encoders:
                for keep_top in keep_tops:
                    for impute_cat in impute_cats:
                        for autofeat in autofeats:
                            for dimreduc in dimreducs:
                                for feature_selector in feature_selectors:
                        
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
                                    data_sheet['autofeat'] = str(autofeat)
                                    data_sheet['dimreduc'] = str(dimreduc)
                                    data_sheet['feature_selector'] = str(feature_selector)

                                    config_summary = "{}_{}_{}_{}_{}_{}_{}_{}".format(str(num_indicator), 
                                                                             str(num_imputer),
                                                                             str(cat_encoder), 
                                                                             keep_top, 
                                                                             impute_cat,
                                                                             str(autofeat),
                                                                             str(dimreduc),
                                                                             str(feature_selector)
                                                                            )

                                    data_sheet['config_summary'] = config_summary

                                    top_n_values = {}

                                    hack_args = {
                                        "target_col": target_col,
                                        "id_col": id_col,
                                        "num_imputer": num_imputer,
                                        "num_indicator": num_indicator,
                                        "enc": cat_encoder,
                                        "top_n_values": top_n_values,
                                        "impute_cat": impute_cat,
                                        "keep_top": keep_top,
                                        "autofeat": autofeat,
                                        "dimreduc": dimreduc,
                                        "feature_selector": feature_selector,
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

                                    fn_train = get_fn(train_input, data_dir, data_id)
                                    X_train.to_csv(fn_train, index=False)
                                    print("DEBUG: Wrote file {}".format(fn_train))

                                    fn_test = get_fn(test_input, data_dir, data_id)
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
