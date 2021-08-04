import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import os
import sys

import json
import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype
from autofeat import AutoFeatRegressor, AutoFeatClassifier, AutoFeatLight

from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, MissingIndicator
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import StratifiedKFold
import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper

import geopy.distance


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dump_json(fn, json_obj):
    # Write json_obj to a file named fn

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

    with open(fn, 'w') as fp:
        json.dump(json_obj, fp, indent=4, cls=NumpyEncoder)

class SteveCorexWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, bin_cols=None, drop_orig=False, n_hidden=4, dim_hidden=2, smooth_marginals=False):
        from bio_corex import corex
        self.bin_cols = bin_cols
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.smooth_marginals = smooth_marginals
        self.corex = corex.Corex(n_hidden=n_hidden, dim_hidden=dim_hidden, marginal_description='discrete', smooth_marginals=smooth_marginals)
        self.drop_orig = drop_orig

    def fit(self, X, y=None):
        if self.bin_cols:
            self.corex.fit(X[self.bin_cols])
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        _new_cols = self.corex.transform(_X[self.bin_cols])
        _new_cols = pd.DataFrame(_new_cols, columns=["{}_corex".format(c) for c in range(self.n_hidden)])
        _X = pd.concat([_X, _new_cols], axis=1, ignore_index=False)
        if self.drop_orig:
            _X = _X.drop(self.bin_cols, axis=1)
        return _X

class SteveCategoryCoalescer(BaseEstimator, TransformerMixin):
    #Coalesces (smushes) rare levels in a categorical feature into "__OTHER__"
    def __init__(self, keep_top=25, cat_cols=None, id_col_name=''):
        self.keep_top = keep_top
        self.id_col_name = id_col_name
        
        # For each cat_col, this will hold an list of the top values
        self.top_n_values = {}
       
        self.cat_cols = cat_cols
            
    def get_top_n_values(self, X, column, start_list=[], n=5):

        top_values = start_list

        vc = X[column].value_counts(sort=True, ascending=False)
        vals = list(vc.index)
        if len(vals) > n:
            top_values = top_values + vals[0:n]
        else:
            top_values = top_values + vals
        return top_values
    
    def keep_only(self, X, column, keep_list, replace_val='__OTHER__'):
        X.loc[~X[column].isin(keep_list), column] = replace_val
        return X

    def fit(self, X, y=None):
        
        if self.cat_cols is None:
            self.cat_cols = []
            for c in X.columns:
                if c == self.id_col_name:
                    continue
                col_type = X[c].dtype
                if col_type == 'object' or col_type.name == 'category':
                    self.cat_cols.append(c)
                    
        for c in self.cat_cols:
            self.top_n_values[c] = self.get_top_n_values(X, c, n=self.keep_top)
        return self
    
    def transform(self, X, y=None):
        _X = X.copy()
        for c in self.cat_cols:
            _X.loc[~_X[c].isin(self.top_n_values[c]), c] = "__OTHER__"
        return _X

    
class SteveEQPrepper(BaseEstimator, TransformerMixin):
    def __init__(self, min_num=25):
        #print("SteveEQPrepper: init")
        self.min_num = min_num
        self.level_3_counts = None

    def fit(self, X, y=None):
        #print("SteveEQPrepper: fit")
        self.level_3_counts = X.groupby('geo_level_3_id').agg({'geo_level_3_id': 'count'})
        return self

    def transform(self, X, y=None):
        #print("SteveEQPrepper: tranform")
        _X = X.copy()

        def f(row, level_3_counts, min_num):
            geo_level_1_id = row['geo_level_1_id']
            geo_level_2_id = row['geo_level_2_id']
            geo_level_3_id = row['geo_level_3_id']
            return int(geo_level_3_id)
            if geo_level_3_id not in level_3_counts.index or level_3_counts.loc[geo_level_3_id][0] < min_num:
                return geo_level_2_id
            else:
                return geo_level_3_id

        _X['geo_level_3_id'] = _X.apply(f, args=(self.level_3_counts, self.min_num), axis=1)

        return _X
    
    
class SteveCategoryImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols=None):
        self.cat_cols = cat_cols
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        _X = X.copy()
        
        # TODO: not really doing anything here
        _X[self.cat_cols] = _X[self.cat_cols].astype('category')
        
        for cat_col in self.cat_cols:
            _X[cat_col] = _X[cat_col].cat.add_categories('__NAN__')
        _X[self.cat_cols] = _X[self.cat_cols].fillna('__NAN__')
        _X[self.cat_cols] = _X[self.cat_cols].astype('category')
        
        return _X
    
 
class SteveNumericNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, float_cols=None, drop_orig=True):
        self.float_cols = float_cols
        self.drop_orig = drop_orig
        self.scaler = StandardScaler()
        self.kbins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.float_cols])
        self.kbins.fit(X[self.float_cols])
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        
        _new_cols = self.scaler.transform(_X[self.float_cols])
        _new_cols = pd.DataFrame(_new_cols, columns=["{}_scale".format(c) for c in self.float_cols])
        _X = pd.concat([_X, _new_cols], axis=1, ignore_index=False)
        
        _new_cols = self.kbins.transform(_X[self.float_cols])
        _new_cols = pd.DataFrame(_new_cols, columns=["{}_kbins".format(c) for c in self.float_cols])
        _X = pd.concat([_X, _new_cols], axis=1, ignore_index=False)
        
        if self.drop_orig:
            _X = _X.drop(self.float_cols, axis=1)
        return _X
    
class SteveDateExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols=None, drop_orig=True):
        self.date_cols = date_cols
        self.drop_orig = drop_orig
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        _X = X.copy()
        
        for col in self.date_cols:
            _X[col] =  pd.to_datetime(_X[col], errors='coerce')
            tmp_dt = _X[col].dt
            new_columns_dict = {f'year_{col}': tmp_dt.year, 
                                f'month_{col}': tmp_dt.month,
                                f'day_{col}': tmp_dt.day, 
                                #f'hour_{col}': tmp_dt.hour,
                                #f'minute_{col}': tmp_dt.minute, 
                                #f'second_{col}': tmp_dt.second,
                                f'dayofweek_{col}': tmp_dt.dayofweek,
                                f'dayofyear_{col}': tmp_dt.dayofyear,
                                f'quarter_{col}': tmp_dt.quarter}
            for new_col_name in new_columns_dict.keys():
                if new_col_name not in _X.columns and \
                        new_columns_dict.get(new_col_name).nunique(dropna=False) >= 2:
                    _X[new_col_name] = new_columns_dict.get(new_col_name)
                    
        if self.drop_orig:
            _X = _X.drop(self.date_cols, axis=1)
        
        return _X
    
class SteveLatLongDist(BaseEstimator, TransformerMixin):
    def __init__(self, lat_col=None, long_col=None, point_list=[]):
        self.lat_col = lat_col
        self.long_col = long_col
        self.point_list = point_list
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        _X = X.copy()
        
        for point in self.point_list:
            point_name = point[0] # Should be a string
            point_coords = point[1] #Should be a 2-typle
            
            _X['{}_dist'.format(point_name)] = _X[[self.lat_col, self.long_col]].apply(lambda x: geopy.distance.great_circle(point_coords, (x[0], x[1])).km, axis=1)
        return _X
    
class SteveMissingIndicator(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=[]):
        self.num_cols = num_cols
        self.num_indicator = MissingIndicator(features="all")
    def fit(self, X, y=None):
        self.num_indicator.fit(X[self.num_cols], y)
        return self
    def transform(self, X, y=None):
        _X = X.copy()
    
        _new_cols = self.num_indicator.transform(_X[self.num_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_cols = pd.DataFrame(_new_cols, columns=["missing_{}".format(c) for c in self.num_cols])
        _X = pd.concat([_X, _new_cols], axis=1, ignore_index=False)
        
        return _X
    
class SteveAutoFeatLight(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=[], compute_ratio=False, compute_product=True, scale=False):
        self.num_cols = num_cols
        self.compute_ratio = compute_ratio
        self.compute_product = compute_product
        self.scale = scale
        self.autofeat = AutoFeatLight(verbose=0, compute_ratio=compute_ratio, compute_product=compute_product, scale=scale)
        
    def fit(self, X, y=None):
        self.autofeat.fit(X[self.num_cols])
        return self
    
    def transform(self, X, y=None):
        _X = X.copy()
    
        _new_cols = self.autofeat.transform(_X[self.num_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_cols = pd.DataFrame(_new_cols, columns=["autofeat_{}".format(c) for c in self.num_cols])
           
        # Autofit will create duplicate columns - remove
        _new_cols = _new_cols.drop(self.num_cols, axis=1)
        _X = pd.concat([_X, _new_cols], axis=1, ignore_index=False)
        
        return _X
        
   
def get_data_types(df, id_col, target_col):
    ##################################################
    # Gather data types
    ##################################################
    def is_binary(series):
        #series.dropna(inplace=True)
        return series.dropna().nunique() <= 2
    
    cat_cols = []
    num_cols = []
    float_cols = []
    bin_cols = []
    date_cols = []
    
    for c in df.columns:
        if c == id_col or c == target_col:
            continue
        col_type = df[c].dtype
        if 'datetime' in col_type.name:
            date_cols.append(c)
        elif col_type == 'object' or col_type.name == 'category':
            cat_cols.append(c)
        elif is_numeric_dtype(df[c]):
            num_cols.append(c)
        if is_binary(df[c]):
            bin_cols.append(c)
            
        if is_numeric_dtype(df[c]) and not is_binary(df[c]) and c not in cat_cols:
            float_cols.append(c)
            
    print("DEBUG: Cat cols: {}".format(cat_cols))
    print("DEBUG: Num cols: {}".format(num_cols))
    print("DEBUG: Bin cols: {}".format(bin_cols))
    print("DEBUG: Float cols: {}".format(float_cols))
    print("DEBUG: Date cols: {}".format(date_cols))
    
    return cat_cols, num_cols, bin_cols, float_cols, date_cols