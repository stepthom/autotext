import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import os
import sys
import re

import time

import json
import datetime

import scipy.stats

from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype
from autofeat import AutoFeatRegressor, AutoFeatClassifier, AutoFeatLight

from sklearn.pipeline import Pipeline
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
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

import geopy.distance

class StudyData:
    def __init__(self, study, sample_frac=1.0):
        
        self.train_df = pd.read_csv(study.user_attrs['train_fn'])
        self.test_df = pd.read_csv(study.user_attrs['test_fn'])
        if sample_frac< 1.0:
            self.train_df  = self.train_df.sample(frac=sample_frac, random_state=3).reset_index(drop=True)

        self.X = self.train_df.drop([study.user_attrs['id_col'], study.user_attrs['target_col']], axis=1)
        self.X_test = self.test_df.drop([study.user_attrs['id_col']], axis=1)
        self.y = self.train_df[study.user_attrs['target_col']]
        self.label_transformer = LabelEncoder()
        self.y = self.label_transformer.fit_transform(self.y)
        

def get_pipeline_steps(study_name, pipe_args):
    # Have to create a new pipeline all the time, thanks to a bug in category_encoders:
    # https://github.com/scikit-learn-contrib/category_encoders/issues/313
    
    steps = []
    
    if study_name == "h1n1":
    
        _drop_cols = pipe_args.get('drop_cols', [])
        if len(_drop_cols) > 0:
            steps.append(('ddropper', SteveFeatureDropper(_drop_cols)))

        # Indicator num
        _num_cols = pipe_args.get('num_cols_indicator', [])
        if len(_num_cols) > 0:
            steps.append(
                ('num_indicator', 
                 SteveMissingIndicator(num_cols=_num_cols)
                )
            )
            #steps.append(('bool_typer', SteveFeatureTyper(like="_missing", typestr='int32')))

        # Impute num
        _num_cols = pipe_args.get('num_cols_impute', [])
        if len(_num_cols) > 0:
            steps.append(
                ('num_imputer', 
                 SteveNumericImputer(num_cols=_num_cols, imputer=SimpleImputer(missing_values=np.nan, strategy="median"))
                )
            )

        # Cat impute
        _cat_cols = pipe_args.get('cat_cols_impute', [])
        if len(_cat_cols) > 0:
            steps.append(('cat_imputer', SteveCategoryImputer(_cat_cols)))

        # Cat smush
        _cat_cols = pipe_args.get('cat_cols_smush', [])
        if len(_cat_cols) > 0:
            steps.append(('cat_smush', SteveCategoryCoalescer(keep_top=5, cat_cols=_cat_cols)))

        # Cat encode
        _cat_cols = pipe_args.get('cat_cols_ordinal_encode', [])
        if len(_cat_cols) > 0:
            steps.append(
                ('cat_encoder', 
                 SteveEncoder(
                     cols=_cat_cols,
                     encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32),
                     suffix="_oenc"
                 )))
            steps.append(('odropper', SteveFeatureDropper(_cat_cols)))


        _cat_cols = pipe_args.get('cat_cols_onehot_encode', [])
        if len(_cat_cols) > 0:
            steps.append(
                ('cat_enc1', 
                 SteveEncoder(
                     cols=_cat_cols,
                     encoder=OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int32),
                     suffix="_oheenc")
                ))
            steps.append(('phedropper', SteveFeatureDropper(_cat_cols)))

        _cat_cols = pipe_args.get('cat_cols_target_encode', [])
        if len(_cat_cols) > 0:
            steps.append(('typer', SteveFeatureTyper(cols=_cat_cols, typestr='category')))
            enc =  ce.wrapper.PolynomialWrapper(
                    ce.target_encoder.TargetEncoder(
                        handle_unknown="value", 
                        handle_missing="value", 
                        min_samples_leaf=1, 
                        smoothing=0.1, return_df=True))

            steps.append(
                ('cat_enc2', 
                 SteveEncoder( cols=_cat_cols, encoder=enc, suffix="_tenc"
                 )))

            steps.append(('tdropper', SteveFeatureDropper(_cat_cols)))

        _float_cols = pipe_args.get('float_cols', [])
        if len(_float_cols) > 0 and pipe_args.get('autofeat', 0) == 1:
            steps.append(('num_autofeat', SteveAutoFeatLight(_float_cols, compute_ratio=True, compute_product=True, scale=True)))

        if len(_float_cols) > 0 and pipe_args.get('normalize', 0) == 1:
            steps.append(('num_normalizer', SteveNumericNormalizer(_float_cols, drop_orig=True)))
        
    return steps

def run_one(study_data, study_name, pipe_args, estimator):
    """
    X, y: features and target
    pipe_args: args to control FE pipeline
    estimator: the estmator
    
    Returns metrics, such as:
    val_scores: estimated val score for each each CV fold
    train_scores: estimated val score for each each CV fold
    """
    
    print("run_one: pipe_args: {}".format(pipe_args))
          
    steps = get_pipeline_steps(study_name, pipe_args)
    pipe = Pipeline(steps)

    pipe.fit(X, y)
    _X = pipe.transform(X)
    check_dataframe(_X, "_X")
        
    extra_fit_params = {}
    if isinstance (estimator, LGBMClassifier) or isinstance(estimator, HistGradientBoostingClassifier) or isinstance(estimator, XGBClassifier):
        indices =  [i for i, ix in enumerate(_X.columns.values) if "_oenc" in ix]
        if len(indices) > 0:
            if isinstance (estimator, LGBMClassifier):
                extra_fit_params.update({
                    'categorical_feature': indices, 
                })
            elif isinstance (estimator, HistGradientBoostingClassifier):
                estimator.set_params(**{
                    'categorical_features': indices, 
                })
    print("run_one: extra_fit_params:")
    print(extra_fit_params)

    start = time.time()
    print("run_one: estimator: {}".format(estimator))
    print("run_one: fitting...: ")
    estimator.fit(_X, y, **extra_fit_params)
    duration = (time.time() - start)
    print("run_one: done. Time {}".format(duration))
    
    return pipe, estimator


def estimate_metrics(study_data, study_name, pipe_args, estimator, num_cv=5, early_stop=True, metric="f1"):
    """
    X, y: features and target
    pipe_args: args to control FE pipeline
    estimator: the estmator
    use early stopping (for lgbm and xgboost)?
    
    Returns metrics, such as:
    val_scores: estimated val score for each each CV fold
    train_scores: estimated val score for each each CV fold
    """
    metrics = {}
    val_scores = []
    train_scores = []
    train_times = []
    best_iterations = []
    
    print("estimate_metrics: pipe_args: {}".format(pipe_args))
          
    skf = StratifiedKFold(n_splits=num_cv, random_state=42, shuffle=True)

    cv_step = -1
    for train_index, val_index in skf.split(study_data.X, study_data.y):
        cv_step = cv_step + 1
        print("========================================")
        print("estimate_metrics: cv_step {} of {}".format(cv_step, num_cv))

        X_train, X_val = study_data.X.loc[train_index].reset_index(drop=True), study_data.X.loc[val_index].reset_index(drop=True)
        y_train, y_val = study_data.y[train_index], study_data.y[val_index]

        steps = get_pipeline_steps(study_name, pipe_args)
        pipe = Pipeline(steps)

        pipe.fit(X_train, y_train)

        _X_train = pipe.transform(X_train)
        _X_val  = pipe.transform(X_val)

        #check_dataframe(_X_train, "_X_train", full=False)
        #check_dataframe(_X_val, "_X_val", full=False)
        
        extra_fit_params = {}
        if early_stop and isinstance (estimator, LGBMClassifier) or isinstance(estimator, XGBClassifier):
            extra_fit_params.update({
                'eval_set': [(_X_val, y_val)],
                'early_stopping_rounds': 50,
                'verbose': 200,
            })
        if isinstance (estimator, LGBMClassifier) or isinstance(estimator, HistGradientBoostingClassifier) or isinstance(estimator, XGBClassifier):
            indices =  [i for i, ix in enumerate(_X_train.columns.values) if "_oenc" in ix]
            if len(indices) > 0:
                print("estimate_metric: categorical indices: {}".format(indices))
                if isinstance (estimator, LGBMClassifier):
                    extra_fit_params.update({
                        'categorical_feature': indices, 
                    })
                elif isinstance (estimator, HistGradientBoostingClassifier):
                    estimator.set_params(**{
                        'categorical_features': indices, 
                    })
                elif isinstance (estimator, XGBClassifier):
                    # This appears to not be working; xgboost complains.
                    # Bummber!
                    #estimator.set_params(**{
                        #'enable_categorical': True, 
                    #})
                    pass

        start = time.time()
        print("estimate_metrics: estimator: {}".format(estimator))
        print("estimate_metric: fitting...: ")
        estimator.fit(_X_train, y_train, **extra_fit_params)
        train_times.append((time.time() - start))

        print("estimate_metric: calc Val metrics. ")
        y_val_pred_proba = estimator.predict_proba(_X_val)
        y_val_pred = estimator.predict(_X_val)
        if metric == "f1":
            val_score =  f1_score(y_val, y_val_pred, average="micro")
        elif metric == "roc_auc":
            print('calculating roc_auc')
            val_score =  roc_auc_score(y_val, y_val_pred_proba[:,1], average="macro")
                
        val_scores.append(val_score)
        print("val_score = {}".format(val_score))
        print(classification_report(y_val, y_val_pred, digits=4))

        print("estimate_metric: calc Train metrics. ")
        y_train_pred_proba = estimator.predict_proba(_X_train)
        y_train_pred = estimator.predict(_X_train)
        if metric == "f1":
            train_score =  f1_score(y_train, y_train_pred, average="micro")
        elif metric == "roc_auc":
            train_score =  roc_auc_score(y_train, y_train_pred_proba[:,1], average="macro")
        train_scores.append(train_score)
        #print(classification_report(y_train, y_train_pred, digits=4))

        bi = None
        if hasattr(estimator, 'best_iteration_'):
            bi = estimator.best_iteration_
        elif hasattr(estimator, 'best_iteration'):
            bi = estimator.best_iteration
        if bi is not None:
            best_iterations.append(bi)
          
    def mean_confidence_interval(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, m-h, m+h

    metrics['val_scores'] = val_scores
    metrics['val_scores_range'] = mean_confidence_interval(val_scores)
    metrics['train_scores'] = train_scores
    bir = []
    if len(best_iterations) > 0:
        bir = mean_confidence_interval(best_iterations)
    metrics['best_iterations'] = best_iterations
    metrics['best_iterations_range'] = bir
    metrics['train_times'] = train_times
    
    print("estimate_metrics: cv complete.")
    print("estimate_metrics: metrics:")
    print(metrics)
          
    return metrics
    


def read_json(fn):
    run = {}
    with open(fn) as f:
        try:
            run = json.load(f)
        except JSONDecodeError as e:
            print("ERROR: cannot parse json file {}".format(fn))
            print(e)
    return run


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
        
def check_array(array, name="array", full=True):
    print("=============================")
    print("{} shape: {}".format(name, array.shape))
    print(array[0:5, :])
    print(array[-5:-1, :])
    has_a_inf = np.isinf(array).any().any()
    print("Any inf?: {}".format(has_a_inf))
    has_a_nan = np.isnan(array).any().any()
    print("Any nan?: {}".format(has_a_nan))
    #print("nan: {}".format(np.argwhere(np.isnan(array))))
    print(array[17230,:])
        
def check_dataframe(df, name="df", full=True):
    print("=============================")
    print("{} shape: {}".format(name, df.shape))
    if full:
        print("columns: {}".format(list(df.columns)))
        #print(df.info())
        #print(df.head())
     
    #has_a_inf = np.isinf(df).any().any()
    #print("Any inf?: {}".format(has_a_inf))
    has_a_nan = df.isnull().any().any()
    print("Any nan?: {}".format(has_a_nan))
    if has_a_nan:
        tmp = df.isnull().sum(axis=0)
        tmp = tmp[tmp>0]
        print("{} cols with nan:".format(len(tmp)))
        print(tmp[tmp>0])
        print("Number of rows with nan: {}".format(df.isnull().sum(axis = 1).sum()))
    #print("Any duplicates?: {}".format(df.duplicated().any()))
    dup_mask = df.duplicated(keep=False)
    print("Number of duplicates: {}".format(dup_mask.sum()))
    if dup_mask.sum() > 0:
        print("First 20 duplicate rows:")
        print(df[dup_mask].sort_values(by=list(df.columns),axis=0).head())
        
    
    

class SteveCorexWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, bin_cols=None, drop_orig=False, n_hidden=4, dim_hidden=2, smooth_marginals=False):
        from bio_corex import corex
        self.bin_cols = bin_cols
        self.n_hidden = n_hidden
        self.dim_hidden = dim_hidden
        self.smooth_marginals = smooth_marginals
        self.drop_orig = drop_orig
        self.corex = corex.Corex(n_hidden=n_hidden, dim_hidden=dim_hidden, marginal_description='discrete', smooth_marginals=smooth_marginals)
        
        #self.corex1 = corex.Corex(n_hidden=8, dim_hidden=dim_hidden, marginal_description='discrete', smooth_marginals=smooth_marginals)
        #self.corex2 = corex.Corex(n_hidden=4, dim_hidden=dim_hidden, marginal_description='discrete', smooth_marginals=smooth_marginals)

    def fit(self, X, y=None):
        if self.bin_cols:
            self.corex.fit(X[self.bin_cols])
            #Y1 = self.corex1.fit_transform(X[self.bin_cols])
            #self.corex2.fit(Y1)
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        #Y1 = self.corex1.transform(_X[self.bin_cols])
        #_new_cols = self.corex2.transform(Y1)
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
        _X[self.cat_cols] = _X[self.cat_cols].astype('category')
        for c in self.cat_cols:
            _X[c] = _X[c].cat.add_categories('__OTHER__')
            _X.loc[~_X[c].isin(self.top_n_values[c]), c] = "__OTHER__"
        return _X

    
def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    for idx, dt in enumerate(df.dtypes):
        if dt == 'object' or pd.api.types.is_categorical_dtype(dt):
            obj_cols.append(df.columns.values[idx])

    return obj_cols

def convert_cols_to_list(cols):
    if isinstance(cols, pd.Series):
        return cols.tolist()
    elif isinstance(cols, np.ndarray):
        return cols.tolist()
    elif np.isscalar(cols):
        return [cols]
    elif isinstance(cols, set):
        return list(cols)
    elif isinstance(cols, tuple):
        return list(cols)
    elif pd.api.types.is_categorical(cols):
        return cols.astype(object).tolist()

    return cols

def convert_input(X, columns=None, deep=False):
    """
    Unite data into a DataFrame.
    Objects that do not contain column names take the names from the argument.
    Optionally perform deep copy of the data.
    """
    if not isinstance(X, pd.DataFrame):
        if isinstance(X, pd.Series):
            X = pd.DataFrame(X, copy=deep)
        else:
            if columns is not None and np.size(X,1) != len(columns):
                raise ValueError('The count of the column names does not correspond to the count of the columns')
            if isinstance(X, list):
                X = pd.DataFrame(X, columns=columns, copy=deep)  # lists are always copied, but for consistency, we still pass the argument
            elif isinstance(X, (np.generic, np.ndarray)):
                X = pd.DataFrame(X, columns=columns, copy=deep)
            elif isinstance(X, csr_matrix):
                X = pd.DataFrame(X.todense(), columns=columns, copy=deep)
            else:
                raise ValueError('Unexpected input type: %s' % (str(type(X))))
    elif deep:
        X = X.copy(deep=True)

    return X


def convert_input_vector(y, index):
    """
    Unite target data type into a Series.
    If the target is a Series or a DataFrame, we preserve its index.
    But if the target does not contain index attribute, we use the index from the argument.
    """
    if y is None:
        raise ValueError('Supervised encoders need a target for the fitting. The target cannot be None')
    if isinstance(y, pd.Series):
        return y
    elif isinstance(y, np.ndarray):
        if len(np.shape(y))==1:  # vector
            return pd.Series(y, name='target', index=index)
        elif len(np.shape(y))==2 and np.shape(y)[0]==1:  # single row in a matrix
            return pd.Series(y[0, :], name='target', index=index)
        elif len(np.shape(y))==2 and np.shape(y)[1]==1:  # single column in a matrix
            return pd.Series(y[:, 0], name='target', index=index)
        else:
            raise ValueError('Unexpected input shape: %s' % (str(np.shape(y))))
    elif np.isscalar(y):
        return pd.Series([y], name='target', index=index)
    elif isinstance(y, list):
        if len(y)==0 or (len(y)>0 and not isinstance(y[0], list)): # empty list or a vector
            return pd.Series(y, name='target', index=index, dtype=float)
        elif len(y)>0 and isinstance(y[0], list) and len(y[0])==1: # single row in a matrix
            flatten = lambda y: [item for sublist in y for item in sublist]
            return pd.Series(flatten(y), name='target', index=index)
        elif len(y)==1 and len(y[0])==0 and isinstance(y[0], list): # single empty column in a matrix
            return pd.Series(y[0], name='target', index=index, dtype=float)
        elif len(y)==1 and isinstance(y[0], list): # single column in a matrix
            return pd.Series(y[0], name='target', index=index, dtype=type(y[0][0]))
        else:
            raise ValueError('Unexpected input shape')
    elif isinstance(y, pd.DataFrame):
        if len(list(y))==0: # empty DataFrame
            return pd.Series(name='target', index=index, dtype=float)
        if len(list(y))==1: # a single column
            return y.iloc[:, 0]
        else:
            raise ValueError('Unexpected input shape: %s' % (str(y.shape)))
    else:
        return pd.Series(y, name='target', index=index)  # this covers tuples and other directly convertible types
    
class SteveGeoTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, upper_col=None, cols=None, min_samples_leaf=1, smoothing=1.0):
        #print("SteveEQPrepper: init")
        self.upper_col = upper_col
        self.cols = cols
        self.ordinal_encoder = None
        self.min_samples_leaf = min_samples_leaf
        self.smoothing = float(smoothing)  # Make smoothing a float so that python 2 does not treat as integer division
        self._dim = None
        self.mapping = None
        self.handle_unknown = 'value'
        self.handle_missing = 'value'
        self._mean = None
        self.feature_names = None
        self.verbose = 1

    def fit(self, X, y=None):
        #print("SteveEQPrepper: fit")
        #self.level_3_counts = X.groupby('geo_level_3_id').agg({'geo_level_3_id': 'count'})
        
        X = convert_input(X)
        y = convert_input_vector(y, X.index)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        self._dim = X.shape[1]
        
        
        # if columns aren't passed, just use every string column
        if self.cols is None:
            self.cols = get_obj_cols(X)
        else:
            self.cols = convert_cols_to_list(self.cols)
            
        print("cols = {}".format(self.cols))

        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        self.ordinal_encoder = ce.OrdinalEncoder(
            cols=self.cols,
            handle_unknown='value',
            handle_missing='value'
        )
        self.ordinal_encoder = self.ordinal_encoder.fit(X)
        X_ordinal = self.ordinal_encoder.transform(X)
        self.mapping = self.fit_target_encoding(X_ordinal, y)
        
        X_temp = self.transform(X)
        self.feature_names = list(X_temp.columns)

        return self
    
    
    def fit_target_encoding(self, X, y):
        mapping = {}

        for switch in self.ordinal_encoder.category_mapping:
            col = switch.get('col')
            values = switch.get('mapping')
            
            _X = X.copy()
            _X['__target__'] = y

            #self.level_3_counts = X.groupby('geo_level_3_id').agg({'geo_level_3_id': 'count'})
            #prior = 
            prior = self._mean = y.mean()
            

            stats = y.groupby(X[col]).agg(['count', 'mean'])
            print("")
            print("col={}\n values={}\n prior={}\n stats={}".format(col, values, prior, stats))
            print(y.head())

            smoove = 1 / (1 + np.exp(-(stats['count'] - self.min_samples_leaf) / self.smoothing))
            print("smoove={}".format(smoove))
            smoothing = prior * (1 - smoove) + stats['mean'] * smoove
            print("smoothing={}".format(smoothing))
            smoothing[stats['count'] == 1] = prior
            print("smoothing={}".format(smoothing))

            if self.handle_unknown == 'return_nan':
                smoothing.loc[-1] = np.nan
            elif self.handle_unknown == 'value':
                smoothing.loc[-1] = prior

            if self.handle_missing == 'return_nan':
                smoothing.loc[values.loc[np.nan]] = np.nan
            elif self.handle_missing == 'value':
                smoothing.loc[-2] = prior

            mapping[col] = smoothing

        return mapping

    def transform(self, X, y=None):
        if self.handle_missing == 'error':
            if X[self.cols].isnull().any().any():
                raise ValueError('Columns to be encoded can not contain null')

        if self._dim is None:
            raise ValueError('Must train encoder before it can be used to transform data.')

        # then make sure that it is the right size
        if X.shape[1] != self._dim:
            raise ValueError('Unexpected input dimension %d, expected %d' % (X.shape[1], self._dim,))

        # if we are encoding the training data, we have to check the target
        if y is not None:
            y = convert_input_vector(y, X.index)
            if X.shape[0] != y.shape[0]:
                raise ValueError("The length of X is " + str(X.shape[0]) + " but length of y is " + str(y.shape[0]) + ".")

        if not list(self.cols):
            return X

        X = self.ordinal_encoder.transform(X)

        if self.handle_unknown == 'error':
            if X[self.cols].isin([-1]).any().any():
                raise ValueError('Unexpected categories found in dataframe')

        X = self.target_encode(X)

        return X
    
    def target_encode(self, X_in):
        X = X_in.copy(deep=True)

        for col in self.cols:
            X[col] = X[col].map(self.mapping[col])

        return X
    
    
    
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
    
class SteveEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, encoder=None, suffix="_enc"):
        self.cols = cols
        self.encoder = encoder
        self.suffix = suffix
        
    def fit(self, X, y=None):
        #print('SteveEncoder')
        #print(self.cols)
        #print(self.encoder)
        #print(type(X))
        #print(X.shape)
        self.encoder.fit(X[self.cols], y)
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        #print('steve encoder:trans')
        #print(_X.shape)
        #print(_X.columns)
        
        _new_cols = self.encoder.transform(_X[self.cols])
        
        colnames = ["{}_{}".format(i, self.suffix) for i in range(_new_cols.shape[1])]
        colnames = [re.sub('[^A-Za-z0-9_]+', 'J', x) for x in colnames]
        
        if isinstance(_new_cols, pd.DataFrame):
            _new_cols.columns = colnames
        else:
            _new_cols = pd.DataFrame(_new_cols, columns=colnames)
            
        #print(_new_cols.shape)
        #print(_X.head())
        _X = pd.concat([_X.reset_index(drop=True), _new_cols.reset_index(drop=True)], axis=1, ignore_index=False)
        #print("After concat")
        #print(_X.shape)
        #print(_X.head())
        
        return _X
    
class SteveNumericImputer(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None, imputer=None):
        self.num_cols = num_cols
        self.imputer = imputer
        
    def fit(self, X, y=None):
        print('SteveNumericImputer')
        self.imputer.fit(X[self.num_cols], y)
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        
        _X[self.num_cols] = self.imputer.transform(_X[self.num_cols])
        
        # Preserve data types of original dataframe
        for col in self.num_cols:
            _X[col] =  _X[col].astype(X.dtypes[col])
        
        return _X

class SteveNumericCapper(BaseEstimator, TransformerMixin):
    # Caps a given column to a certain number
    # E.g., any value greater becomes the max
    def __init__(self, num_cols=None, max_val=0):
        self.num_cols = num_cols
        self.max_val = max_val
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        _X = X.copy()
        
        for col in self.num_cols:
            _X[col] =  _X[col].apply(lambda x: x if x <= self.max_val else self.max_val)
        
        return _X
 
class SteveNumericNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, float_cols=None, drop_orig=False):
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
            new_columns_dict = {f'{col}_date_year': tmp_dt.year, 
                                f'{col}_date_month': tmp_dt.month,
                                f'{col}_date_day': tmp_dt.day, 
                                #f'{col}_date_hour': tmp_dt.hour,
                                #f'{col}_date_minute': tmp_dt.minute, 
                                #f'{col}_date_second': tmp_dt.second,
                                f'{col}_date_dayofweek': tmp_dt.dayofweek,
                                f'{col}_date_dayofyear': tmp_dt.dayofyear,
                                f'{col}_date_quarter': tmp_dt.quarter}
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
        #print('SteveMissingIndicator:fit')
        self.num_indicator.fit(X[self.num_cols], y)
        return self
    def transform(self, X, y=None):
        _X = X.copy()
    
        _new_cols = self.num_indicator.transform(_X[self.num_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_cols = pd.DataFrame(_new_cols, columns=["{}_missing".format(c) for c in self.num_cols])
        _X = pd.concat([_X, _new_cols], axis=1, ignore_index=False)
        
        return _X
    
#class SteveGenericTransformerWrapper(BaseEstimator, TransformerMixin):
    #def __init__(self, cols=[], col_types=[], cols_pattern=None, drop_orig=False, transformer=None, prefix="trans_", **init_params):
    
class SteveAutoFeatLight(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=[], compute_ratio=True, compute_product=True, scale=False):
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
        # Autofit will create duplicate columns - remove
        _new_cols = _new_cols.drop(self.num_cols, axis=1, errors='ignore')
       
        print("SteveautoFeatLight: Column names:")
        for i in range(_new_cols.shape[1]):
            print("i={}, col={}".format(i, _new_cols.columns[i]))
            
        # Rename the columns
        _new_cols.columns = ["{}_autofeat".format(i) for i in range(_new_cols.shape[1])]
           
        _X = pd.concat([_X, _new_cols], axis=1, ignore_index=False)
        
        return _X
    
class SteveKernelPCA(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=[], drop_orig=False, sample_frac=None, n_components=20, kernel='rbf', eigen_solver="arpack", max_iter=500):
        self.num_cols = num_cols
        self.drop_orig = drop_orig
        self.sample_frac = sample_frac
        self.pca = KernelPCA(n_components=n_components, kernel=kernel, n_jobs=8, eigen_solver=eigen_solver, max_iter=max_iter)
        
    def fit(self, X, y=None):
        _X = X.copy()
        if self.sample_frac is not None:
            _X = _X.sample(frac=self.sample_frac, replace=False, random_state=42, axis=0)
        self.pca.fit(_X[self.num_cols])
        return self
    
    def transform(self, X, y=None):
        _X = X.copy()
        
        _new_cols = self.pca.transform(_X[self.num_cols])
        if not isinstance(_new_cols, pd.DataFrame):
            _new_col_names =  ["{}_{}".format(i, 'pca') for i in range(_new_cols.shape[1])]
            _new_cols = pd.DataFrame(_new_cols, columns=_new_col_names)
        _X = pd.concat([_X, _new_cols], axis=1, ignore_index=False)
        
        if self.drop_orig:
            _X = _X.drop(self.num_cols, axis=1)
    
        return _X
    
    
class SteveDateDiffer(BaseEstimator, TransformerMixin):
    # date_col2 is the name of a column (that has a date),
    def __init__(self, date_col1=None, date_col2=None, new_col="timediff"):
        self.date_col1 = date_col1
        self.date_col2 = date_col2
        self.new_col = new_col
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        _X = X.copy()
        _X[self.new_col] = (_X[self.date_col1] - _X[self.date_col2]).dt.days
        return _X
    
class SteveConstantDateDiffer(BaseEstimator, TransformerMixin):
    # date_col2 is a constant datetime object
    def __init__(self, date_col1=None, date2=None, new_col="timediff"):
        self.date_col1 = date_col1
        self.date2 = date2
        self.new_col = new_col
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        _X = X.copy()
        _X[self.new_col] = (_X[self.date_col1] - self.date2).dt.days
        return _X
    
class SteveFeatureDropper(BaseEstimator, TransformerMixin):
    # inverse = True means drop all but
    # "like" will include column names like the pattern
    def __init__(self, cols=[], like=None, inverse=False):
        self.cols = cols
        self.like = like
        self.inverse = inverse
    def fit(self, X, y=None):
        self._cols = self.cols
        if self.like is not None:
            for col in X.columns.values:
                if self.like in col:
                    self._cols.append(col)
            
        self._cols = list(set(self._cols))
        return self
    def transform(self, X, y=None):
        _X = X.copy()
        if self.inverse:
            _X = _X[self._cols]
        else:
            _X = _X.drop(self._cols, axis=1)
        return _X
    
class SteveFeatureTyper(BaseEstimator, TransformerMixin):
    # Change all cols to type type
    def __init__(self, cols=[], like=None, typestr=None):
        self.cols = cols
        self.like = like
        self.typestr = typestr
    def fit(self, X, y=None):
        #print("SteveFeatureTyper:fit")
        self._cols = self.cols
        if self.like is not None:
            for col in X.columns.values:
                if self.like in col:
                    self._cols.append(col)
        self._cols = list(set(self._cols))
        return self
    def transform(self, X, y=None):
        #print("SteveFeatureTyper:transform")
        _X = X.copy()
        _X[self._cols] = _X[self._cols].astype(self.typestr)
        return _X
    
class SteveValueReplacer(BaseEstimator, TransformerMixin):
    # Change all cols to type type
    def __init__(self, cols=[], to_replace=None, value=None):
        self.cols = cols
        self.to_replace = to_replace
        self.value = value
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        _X = X.copy()
        _X[self.cols] = _X[self.cols].replace(self.to_replace, self.value)
        return _X
    
class SteveMeansByColValue(BaseEstimator, TransformerMixin):
    # For the given column of interest (categorical),
    # will compute the means of other columsn for each level
    # and add a new feature
    def __init__(self, col_of_interest=None, num_cols=[]):
        self.col_of_interest = col_of_interest
        self.num_cols = num_cols
        self.val_by_col_of_interest = {}
    def fit(self, X, y=None):
        for c in self.num_cols:
            mean_val = X[c].mean(skipna=True)
            self.val_by_col_of_interest[c] =  X.groupby(self.col_of_interest).agg({c: lambda x: x.mean(skipna=True)}).fillna(mean_val)
            
        return self
    def transform(self, X, y=None):
        _X = X.copy()
        for c in self.num_cols:
            _val_by_region = self.val_by_col_of_interest[c]
            _X["{}_{}_mean".format(c, self.col_of_interest)] = _X[self.col_of_interest].map(_val_by_region[c])

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
    
    for c in df.columns.values:
        if c == id_col or c == target_col:
            continue
        col_type = df[c].dtype
        if 'datetime' in col_type.name:
            date_cols.append(c)
        elif col_type == 'object' or col_type.name == 'category':
            cat_cols.append(c)
        elif is_numeric_dtype(df[c]):
            num_cols.append(c)
        if is_binary(df[c]) and is_numeric_dtype(df[c]):
            bin_cols.append(c)
            
        if is_numeric_dtype(df[c]) and not is_binary(df[c]) and c not in cat_cols:
            float_cols.append(c)
            
    print("DEBUG: Cat cols: {}".format(cat_cols))
    print("DEBUG: Num cols: {}".format(num_cols))
    print("DEBUG: Bin cols: {}".format(bin_cols))
    print("DEBUG: Float cols: {}".format(float_cols))
    print("DEBUG: Date cols: {}".format(date_cols))
    
    return cat_cols, num_cols, bin_cols, float_cols, date_cols