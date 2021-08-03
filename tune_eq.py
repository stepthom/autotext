import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import uuid
import argparse
import numpy as np
import pandas as pd
import os
import sys

import json
import socket
import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from pandas.api.types import is_numeric_dtype


from flaml import AutoML

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

        
"""
class SteveCategoryCoalescer(BaseEstimator, TransformerMixin):
    #Coalesces (smushes) rare levels in a categorical feature into "__OTHER__"
    def __init__(self, keep_top=25, cat_cols=None, id_col_name=''):
        self.keep_top = keep_top
        self.id_col_name = id_col_name
        
        # For each cat_col, this will hold an list of the top values
        self.top_n_values = {}
       
        self.cat_cols = cols
            
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
            self.top_n_values[c] = get_top_n_values(X, c, n=self.keep_top)
        return self
    
    def transform(self, X, y=None):
        _X = X.copy()
        for c in self.cat_cols:
            _X[c] = _X.loc[~_X[c].isin(self.top_n_values[c]), c] = "__OTHER__"
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
"""
   
"""
class SteveNumericNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None):
        self.num_cols = num_cols
        #self.scaler = StandardScaler()
        #self.kbins = KBinsDiscretizer(n_bins=20, encode='ordinal', strategy='quantile')

    def fit(self, X, y=None):
        self.corex.fit(X[self.num_cols])
        return self

    def transform(self, X, y=None):
        #print("SteveEQPrepper: tranform")
        _X = X.copy()

        return _X
"""
    
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


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--geo-id-set', type=int, default=3)
    parser.add_argument('-a', '--algo-set', type=int, default=1)
    parser.add_argument('--n_hidden', type=int, default=4)
    parser.add_argument('--dim_hidden', type=int, default=2)
    parser.add_argument('--smooth-marginals', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    parser.add_argument('--min-sample-leaf', type=int, default=1)
    parser.add_argument('--smoothing', type=float, default=10.0)
    parser.add_argument('--ensemble', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']))
    
    args = parser.parse_args()
    
    id_col = 'building_id'
    target_col = 'damage_grade'
    out_dir = 'earthquake/out'
    data_id = '000'
    metric = "micro_f1"

    train_df  = pd.read_csv('earthquake/earthquake_train.csv')
    #train_df  = train_df.sample(frac=0.1, random_state=3).reset_index(drop=True)
    test_df  = pd.read_csv('earthquake/earthquake_test.csv')
    
    if args.geo_id_set == 1:
        train_df[['geo_level_1_id']] = train_df[['geo_level_1_id']].astype('category')
    elif args.geo_id_set == 2:
        train_df[['geo_level_1_id']] = train_df[['geo_level_1_id']].astype('category')
        train_df[['geo_level_2_id']] = train_df[['geo_level_2_id']].astype('category')
    else:
        train_df[['geo_level_1_id']] = train_df[['geo_level_1_id']].astype('category')
        train_df[['geo_level_2_id']] = train_df[['geo_level_2_id']].astype('category')
        train_df[['geo_level_3_id']] = train_df[['geo_level_3_id']].astype('category')
        
        
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
    for c in train_df.columns:
        if c == id_col or c == target_col:
            continue
        col_type = train_df[c].dtype
        if col_type == 'object' or col_type.name == 'category':
            cat_cols.append(c)
        elif is_numeric_dtype(train_df[c]):
            num_cols.append(c)
        if is_binary(train_df[c]):
            bin_cols.append(c)
            
        if is_numeric_dtype(train_df[c]) and not is_binary(train_df[c]) and c not in cat_cols:
            float_cols.append(c)
            
    print("DEBUG: Cat cols: {}".format(cat_cols))
    print("DEBUG: Num cols: {}".format(num_cols))
    print("DEBUG: Bin cols: {}".format(bin_cols))
    print("DEBUG: Float cols: {}".format(float_cols))
        
    estimator_list = ['lgbm']
    if args.algo_set == 1:
        estimator_list = ['lgbm']
    elif args.algo_set == 2:
        estimator_list = ['xgboost']
    elif args.algo_set == 3:
        estimator_list = ['lgbm', 'xgboost', 'catboost']
    elif args.algo_set == 4:
        estimator_list = ['catboost']

    X = train_df.drop([id_col, target_col], axis=1)
    y = train_df[target_col]
    X_test = test_df.drop([id_col], axis=1)
    
    from sklearn.pipeline import Pipeline
    
    steps = []
    
    steps.append(('corex', SteveCorexWrapper(bin_cols)))
    #steps.append(('corex', SteveCorexWrapper(bin_cols)))
    
    preprocessor = Pipeline(steps)
    
    preprocessor.fit(X)
    X = preprocessor.transform(X)
    X_test = preprocessor.transform(X_test)
   
    print("X head:", file=sys.stderr)
    print(X.head(), file=sys.stderr)
    print("X_test head:", file=sys.stderr)
    print(X_test.head(), file=sys.stderr)
    
    results = {}
    runname = str(uuid.uuid4())
    run_fn = os.path.join(out_dir, "tune_eq_{}.json".format(runname))
    print("tune_eq: Run name: {}".format(runname))
   
    os.environ['OS_STEVE_MIN_SAMPLE_LEAF'] = str(args.min_sample_leaf)
    os.environ['OS_STEVE_SMOOTHING'] = str(args.smoothing)

    results['runname'] = runname
    results['args'] = vars(args)
    results['hostname'] = socket.gethostname()
    results['starttime'] = str(datetime.datetime.now())

    automl_settings = {
        "time_budget": 50000,
        "log_file_name": "logs/flaml-{}.log".format(runname),
        "task": 'classification',
        "n_jobs": 8,
        "estimator_list": estimator_list,
        "model_history": False,
        "eval_method": "cv",
        "n_splits": 3,
        "metric": metric,
        "log_training_metric": True,
        "verbose": 1,
        "ensemble": args.ensemble,
    }
    clf = AutoML()
    clf.fit(X, y, **automl_settings)

    endtime = str(datetime.datetime.now())
    results['endtime'] = endtime
    results['automl_settings'] =  automl_settings
    results['best_score'] =  1 - clf.best_loss
    results['best_config'] =  clf.best_config
    results['best_estimator'] =  clf.best_estimator
    
    print("Run name: {}".format(runname))
    print("Run file name: {}".format(run_fn))
   
    preds = clf.predict(X_test)
    preds_df = pd.DataFrame(data={'id': test_df[id_col], target_col: preds})
    preds_fn = os.path.join(out_dir, "{}-{}-preds.csv".format(runname, data_id))
    preds_df.to_csv(preds_fn, index=False)
    print("tune_eq: Wrote preds file: {}".format(preds_fn))

    probas = clf.predict_proba(X_test)
    columns = None
    if hasattr(clf, 'classes_'):
        columns = clf.classes_
    probas_df = pd.DataFrame(probas, columns=columns)
    probas_df[id_col] = test_df[id_col]
    probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
    probas_fn = os.path.join(out_dir, "{}-{}-probas.csv".format(runname, data_id))
    probas_df.to_csv(probas_fn, index=False)
    print("tune_eq: Wrote probas file: {}".format(probas_fn))

    results['preds_fn'] = preds_fn
    results['probas_fn'] = probas_fn
    dump_json(run_fn, results)

if __name__ == "__main__":
    main()
