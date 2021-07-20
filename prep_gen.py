import argparse
import numpy as np
import os
import jsonpickle
import json
import datetime
import socket
import pandas as pd
import uuid
from itertools import product

from autofeat import AutoFeatRegressor, AutoFeatClassifier, AutoFeatLight

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer, MissingIndicator
from pandas.api.types import is_numeric_dtype

import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper

import geopy.distance

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

######################################
# Generic helper functions
#######################################

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


######################################
# Custom functions for water pump
#######################################
def pump_datatype_func(df, train=False):
    df['construction_year']= pd.to_numeric(df['construction_year'])
    df['public_meeting'] = df['public_meeting'].astype('str')
    df['permit'] = df['permit'].astype('str')
    df['date_recorded'] = pd.to_datetime(df['date_recorded'])
    return df

def pump_date_func(df, train=False):
    df = add_datetime(df, 'date_recorded')
    baseline = pd.datetime(2014, 1, 1)
    df['date_recorded_since'] = (baseline - df['date_recorded']).dt.days
    df['timediff'] = df['year_date_recorded'] - df['construction_year']
    df = df.drop(['date_recorded'], axis=1)
    return df

def pump_latlong_func(df, train=False):
    daressalaam = (-6.8, 39.283333)
    mwanza = (-2.516667, 32.9)
    arusha = (-3.366667, 36.683333)
    dodoma = (-6.173056, 35.741944)
    df['daressallam_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(daressalaam, (x[0], x[1])).km, axis=1)
    df['mwanza_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(mwanza, (x[0], x[1])).km, axis=1)
    df['arusha_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(arusha, (x[0], x[1])).km, axis=1)
    df['dodoma_dist'] = df[['latitude', 'longitude']].apply(lambda x: geopy.distance.great_circle(dodoma, (x[0], x[1])).km, axis=1)
    return df

def pump_weirdvals_func(df, train=False):
    strzero_cols = ['funder', 'installer']
    for col in strzero_cols:
        df[col] = df[col].replace('0', np.nan)
        
    none_cols = ['wpt_name']
    for col in none_cols:
        df[col] = df[col].replace('none', np.nan)
        
    nan_cols = ['public_meeting', 'permit']
    for col in nan_cols:
        df[col] = df[col].replace('nan', np.nan)
        
    zero_cols = ['amount_tsh', 'population', 'longitude', 'gps_height']
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
            
    df['latitude'] = df['latitude'].round(decimals = 5)
        
    zero_cols = ['amount_tsh', 'population', 'longitude', 'latitude', 'gps_height']
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
    
    return df

def pump_regionmeans_func(df, train=False):
    
    cols = ['gps_height', 'population', 'latitude', 'longitude']
    
    # Static var
    if train and not hasattr(pump_regionmeans_func, "vals_by_region"):
        pump_regionmeans_func.val_by_regions = {}
        for c in cols:
            mean_val = df[c].mean(skipna=True)
            pump_regionmeans_func.val_by_regions[c] =  df.groupby('region_code').agg({c: lambda x: x.mean(skipna=True)}).fillna(mean_val)
            
    for c in cols:
        _val_by_region = pump_regionmeans_func.val_by_regions[c]
        df["{}_regionmean".format(c)] = df['region_code'].map(_val_by_region[c])
        #print(df[['region_code', c,"{}_regionmean".format(c) ]].head())
    
    return df

######################################
# Custom functions for earthquake
#######################################
def earthquake_custom_features_func(df, train=False):
    
    from scipy.stats import zscore
    
    def is_binary(series):
        series.dropna(inplace=True)
        return series.nunique() <= 2
    
    float_cols = [col for col in df if (is_numeric_dtype(df[col]) and not is_binary(df[col]) and col != "building_id")]
    print("DEBUG: earthquake_custom_features: float_cols: {}".format(float_cols))
   
    for col in float_cols:
        # Get zscore
        df['{}_zscore'.format(col)] = zscore(df[col])
        
        # Apply log, but first clip neg numbers to 1, and then add 1
        df['{}_log'.format(col)] = np.log(np.clip(df[col]+1, 1, None))
        
    
    # Static var
    if train and not hasattr(earthquake_custom_features_func, "kbins"):
        earthquake_custom_features_func.kbins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
        earthquake_custom_features_func.kbins = earthquake_custom_features_func.kbins.fit(df[float_cols]) 
        
        
    _new_cols = earthquake_custom_features_func.kbins.transform(df[float_cols])
    if not isinstance(_new_cols, pd.DataFrame):
        _new_cols = pd.DataFrame(_new_cols, columns=["{}_kbins".format(c) for c in float_cols])
    df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
        
    return df


###############################################
# The main preprocessing function
################################################

def hack(X, y=None, 
         train=False, 
         target_col="status_group",
         id_col="id", 
         drop_cols=[],
         custom_begin_funcs=[],
         custom_end_funcs=[],
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
    # Dropping - Won't need at all
    ##################################################
    if drop_cols: 
        print("DEBUG: dropping columns {}".format(drop_cols))
        msgs.append('Dropping cols: {}'.format(drop_cols))
        df = df.drop(drop_cols, axis=1)
        
    ##################################################
    # Custom begin functions
    ##################################################
    for f in custom_begin_funcs:
        print("DEBUG: calling custom function {}".format(f))
        df = f(df, train)
    
    
    ##################################################
    # Gather data types
    ##################################################
    def is_binary(series):
        series.dropna(inplace=True)
        return series.nunique() <= 2
    
    cat_cols = []
    num_cols = []
    float_cols = []
    for c in df.columns:
        if c == id_col:
            continue
        col_type = df[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            cat_cols.append(c)
        elif is_numeric_dtype(df[c]):
            num_cols.append(c)
            
        if is_numeric_dtype(df[c]) and not is_binary(df[c]):
            float_cols.append(c)
            
    msgs.append('Cat cols: {}'.format(cat_cols))
    msgs.append('Num cols: {}'.format(num_cols))
    msgs.append('Float cols: {}'.format(float_cols))
    print("DEBUG: Cat cols: {}".format(cat_cols))
    print("DEBUG: Num cols: {}".format(num_cols))
    print("DEBUG: Float cols: {}".format(float_cols))
    
    all_cols = cat_cols + num_cols
    
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
        print("DEBUG: hack: adding {} cols for missing values".format(_new_cols.shape[1]))
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
    # Custom end functions
    ##################################################
    for f in custom_end_funcs:
        print("DEBUG: calling custom end function {}".format(f))
        df = f(df, train)
        
        
    ##################################################
    # Google-Auto ML-esq features
    ##################################################
    print("DEBUG: hack: scaler/log/kbins on float cols: {}".format(float_cols))
    def my_log(x):
        return np.log(np.clip(x, 1, None))
    
    if train and not hasattr(hack, "scaler"):
        hack.scaler = StandardScaler()
        hack.scaler = hack.scaler.fit(df[float_cols])
        
    if train and not hasattr(hack, "log"):
        hack.log = FunctionTransformer(my_log)
        #hack.log = hack.log.fit(df[float_cols])
        
    if train and not hasattr(hack, "kbins"):
        hack.kbins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')
        hack.kbins = hack.kbins.fit(df[float_cols])
        
    _new_cols = hack.scaler.transform(df[float_cols])
    _new_cols = pd.DataFrame(_new_cols, columns=["{}_scaler".format(c) for c in float_cols])
    df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
    
    _new_cols = hack.log.transform(df[float_cols])
    _new_cols = _new_cols.add_suffix("_log")
    #_new_cols = pd.DataFrame(_new_cols, columns=["{}_log".format(c) for c in float_cols])
    #print(_new_cols)
    df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
   
    _new_cols = hack.kbins.transform(df[float_cols])
    _new_cols = pd.DataFrame(_new_cols, columns=["{}_kbins".format(c) for c in float_cols])
    df = pd.concat([df, _new_cols], axis=1, ignore_index=False)
        

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
        dimreduc_cols = [c for c in num_cols if c in list(df.columns)]
        print("DEBUG: dimensionality reduction on {} cols".format(len(dimreduc_cols)))
        print("DEBUG: dimensionality reduction is {}".format(str(dimreduc)))
        msgs.append("Dimreduc being called")
        _new_cols = None
        if train:
            n_rows = df.shape[0]
            sample_frac = 0.10
            if n_rows > 10000:
                sample_frac=0.02
            elif n_rows < 30000:
                sample_frac=0.3
            _df = df.sample(frac=sample_frac, replace=False, random_state=42, axis=0)
            print("_df shape: {}".format(_df.shape))
            dimreduc = dimreduc.fit(_df[dimreduc_cols])
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
        print("DEBUG: feature_selector on {} cols".format(len(fs_cols)))
        print("DEBUG: feature_selector is {}".format(str(feature_selector)))
        msgs.append("Feature selection being called")
        _new_cols = None
        if train:
            n_rows = df.shape[0]
            sample_frac = 0.50
            if n_rows > 100000:
                sample_frac=0.10
            elif n_rows < 30000:
                sample_frac=0.8
            _df, _, _y, _ = train_test_split(df, y, train_size=sample_frac, random_state=42)
            feature_selector = feature_selector.fit(_df[fs_cols], _y)
        _new_cols = feature_selector.transform(df[fs_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_col_names =  ["{}_{}".format('feature_selector', i) for i in range(_new_cols.shape[1])]
            _new_cols = pd.DataFrame(_new_cols, columns=_new_col_names)
            
        # Be sure to keep index column!
        df = pd.concat([df[[id_col]], _new_cols], axis=1, ignore_index=False)
        

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
   

    drop_colss = [[]]
    custom_begin_funcss = [[]]
    custom_end_funcss = [[]]
    
    num_indicators = [
       MissingIndicator(features="all"),
       #None,
    ]
    
    num_imputers = [
       SimpleImputer(missing_values=np.nan, strategy="median"),
    ]

    cat_encoders = [
        #ce.wrapper.PolynomialWrapper(ce.cat_boost.CatBoostEncoder(handle_unknown="value", sigma=None)),
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32),
        ce.wrapper.PolynomialWrapper(ce.m_estimate.MEstimateEncoder(randomized=False, verbose=0)),
    ]
    
    keep_tops = [25]
    
    impute_cats = [False]
    
    autofeats = [
        None,
        #AutoFeatLight(verbose=1, compute_ratio=False, compute_product=True, scale=False),
    ]
                                              
    dimreducs = [
        KernelPCA(n_components=20, kernel='rbf', n_jobs=10, eigen_solver="arpack", max_iter=500),
        #KernelPCA(n_components=20, kernel='poly', n_jobs=10, eigen_solver="arpack", max_iter=500),
        #KernelPCA(n_components=20, kernel='sigmoid', n_jobs=10, eigen_solver='arpack', max_iter=500),
        None,
        TruncatedSVD(n_components=20, n_iter=5, random_state=42),
    ]
    
    feature_selectors = [
        #SelectFromModel(estimator=ExtraTreesClassifier(n_estimators=100, random_state=42), threshold=-np.inf, max_features=25),
        None,
    ]
    
    if args.competition.startswith("p"):
        train_input = "pump/pump_train.csv"
        test_input = "pump/pump_test.csv"
        data_dir = "pump/data"
        target_col = "status_group"
        id_col = "id"
        out_dir = "pump/out"
       
        drop_colss =  [
            None, 
            #['recorded_by', 'num_private', 'scheme_name', 'payment_type', 
             #'quantity_group', 'source_type', 'waterpoint_type_group']
        ]
        custom_begin_funcss = [[pump_datatype_func, pump_weirdvals_func]]
        custom_end_funcss = [[pump_date_func, pump_latlong_func, pump_regionmeans_func]]
        keep_tops = [25, 200]
        dimreducs = [
            KernelPCA(n_components=15, kernel='rbf', n_jobs=10, eigen_solver="arpack", max_iter=500),
            #KernelPCA(n_components=5, kernel='poly', n_jobs=10, eigen_solver="arpack", max_iter=500),
            #KernelPCA(n_components=5, kernel='sigmoid', n_jobs=10, eigen_solver='arpack', max_iter=500),
            #None,
            #TruncatedSVD(n_components=5, n_iter=5, random_state=42),
        ]
        
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
        test_input = "earthquake/earthquake_test.csv"
        data_dir = "earthquake/data"
        target_col = "damage_grade"
        id_col = "building_id"
        out_dir = "earthquake/out"
        
        num_indicators = [
           MissingIndicator(features="all"),
        ]
        
        cat_encoders = [
            #ce.wrapper.PolynomialWrapper(ce.cat_boost.CatBoostEncoder(handle_unknown="value", sigma=None)),
            #OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32),
            None
        ]
        
        #custom_end_funcss = [[earthquake_custom_features_func]]
        
        dimreducs = [
            None,
            #TruncatedSVD(n_components=25, n_iter=5, random_state=42),
        ]
        
        autofeats = [ None ]
                                              
        feature_selectors = [ None ]
    else:
        print("Error: unknown competition type: {}".format(args.competition))
        return
    
    train_df= pd.read_csv(train_input)
    test_df = pd.read_csv(test_input)
    
    
    all_combos = list(product(drop_colss, custom_begin_funcss, custom_end_funcss, 
                              num_indicators, num_imputers, 
                              cat_encoders, keep_tops, impute_cats, 
                              autofeats, dimreducs, feature_selectors))
   
    i = 0
    for combo in all_combos:
        i = i + 1
        
        drop_cols = combo[0]
        custom_begin_funcs = combo[1]
        custom_end_funcs = combo[2]
        num_indicator = combo[3]
        num_imputer = combo[4]
        cat_encoder = combo[5]
        keep_top = combo[6]
        impute_cat = combo[7]
        autofeat = combo[8]
        dimreduc = combo[9]
        feature_selector = combo[10]
                        
        data_sheet = {}
        data_id = str(uuid.uuid4())
        data_sheet['data_id'] = data_id
        data_sheet['starttime'] = str(datetime.datetime.now())
        data_sheet['hostname'] = socket.gethostname()
        data_sheet['args'] = vars(args)

        data_sheet['drop_cols'] = str(drop_cols)
        data_sheet['custom_begin_funcs'] = str(custom_begin_funcs)
        data_sheet['custom_end_funcs'] = str(custom_end_funcs)
        data_sheet['num_indicator'] = str(num_indicator)
        data_sheet['num_imputer'] = str(num_imputer)
        data_sheet['cat_encoder'] = str(cat_encoder)
        data_sheet['keep_top'] = keep_top
        data_sheet['impute_cat'] = impute_cat
        data_sheet['autofeat'] = str(autofeat)
        data_sheet['dimreduc'] = str(dimreduc)
        data_sheet['feature_selector'] = str(feature_selector)

        config_summary = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
            str(drop_cols), 
            [f.__name__ for f in custom_begin_funcs], 
            [f.__name__ for f in custom_end_funcs], 
            str(num_indicator), 
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
            "drop_cols": drop_cols,
            "custom_begin_funcs": custom_begin_funcs,
            "custom_end_funcs": custom_end_funcs,
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
        
        print("DEBUG: preprocessing combo {} of {}".format(i, len(all_combos)))
        print("DEBUG: config summary : {} ".format(config_summary))

        X_train, train_msgs = hack(train_df.drop(target_col, axis=1), train_df[target_col], train=True, **hack_args)
        X_train[target_col] = train_df[target_col]
        data_sheet['train_msgs'] = train_msgs

        X_test, test_msgs = hack(X=test_df, y=None, train=False, **hack_args)
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
