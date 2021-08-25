import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import os
import sys

import json
import datetime
import time

from xgboost import XGBClassifier
import lightgbm as lgbm
from lightgbm import LGBMClassifier

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
from sklearn.metrics import classification_report, f1_score
import scipy.stats
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
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from SteveHelpers import *

import geopy.distance

def get_pipeline_steps(pipe_args):
    # Have to create a new pipeline all the time, thanks to a bug in category_encoders:
    # https://github.com/scikit-learn-contrib/category_encoders/issues/313
    
    steps = []
    
    _drop_cols = pipe_args.get('drop_cols', [])
    if len(_drop_cols) > 0:
        steps.append(('ddropper', SteveFeatureDropper(_drop_cols)))

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
       
        #steps.append(('otyper', SteveFeatureTyper(like="_oenc", typestr='category')))
        
        
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

    steps.append(('num_capper', SteveNumericCapper(num_cols=['age'], max_val=30)))

    _float_cols = pipe_args.get('float_cols', [])
    if len(_float_cols) > 0 and pipe_args.get('autofeat', 0) == 1:
        steps.append(('num_autofeat', SteveAutoFeatLight(_float_cols, compute_ratio=True, compute_product=True, scale=True)))

    if len(_float_cols) > 0 and pipe_args.get('normalize', 0) == 1:
        steps.append(('num_normalizer', SteveNumericNormalizer(_float_cols, drop_orig=True)))
        
    print('get_pipeline steps:')
    for step in steps:
        print(step)
    return steps


def estimate_metrics(X, y, pipe_args, estimator, num_cv=5, early_stop=True):
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
    for train_index, val_index in skf.split(X, y):
        cv_step = cv_step + 1
        print("estimate_metrics: cv_step {} of {}".format(cv_step, num_cv))

        X_train, X_val = X.loc[train_index].reset_index(drop=True), X.loc[val_index].reset_index(drop=True)
        y_train, y_val = y[train_index], y[val_index]

        steps = get_pipeline_steps(pipe_args)
        pipe = Pipeline(steps)

        pipe.fit(X_train, y_train)

        _X_train = pipe.transform(X_train)
        _X_val  = pipe.transform(X_val)

        check_dataframe(_X_train, "_X_train")
        check_dataframe(_X_val, "_X_val")
        
        extra_fit_params = {}
        if early_stop and isinstance (estimator, LGBMClassifier) or isinstance(estimator, XGBClassifier):
            extra_fit_params.update({
                'eval_set': [(_X_val, y_val)],
                'early_stopping_rounds':20,
                'verbose': 50,
            })
        if isinstance (estimator, LGBMClassifier) or isinstance(estimator, HistGradientBoostingClassifier) or isinstance(estimator, XGBClassifier):
            indices =  [i for i, ix in enumerate(_X_train.columns.values) if "_oenc" in ix]
            if len(indices) > 0:
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
        print("estimate_metric: extra_fit_params:")
        print(extra_fit_params)

        start = time.time()
        print("estimate_metrics: estimator: {}".format(estimator))
        print("estimate_metric: fitting...: ")
        estimator.fit(_X_train, y_train, **extra_fit_params)
        train_times.append((time.time() - start))

        print("estimate_metric: Val: ")
        y_val_pred_proba = estimator.predict_proba(_X_val)
        y_val_pred = estimator.predict(_X_val)
        val_scores.append(f1_score(y_val, y_val_pred, average="micro"))
        print(classification_report(y_val, y_val_pred, digits=4))

        print("estimate_metric: Train: ")
        y_train_pred_proba = estimator.predict_proba(_X_train)
        y_train_pred = estimator.predict(_X_train)
        train_scores.append(f1_score(y_train, y_train_pred, average="micro"))
        print(classification_report(y_train, y_train_pred, digits=4))

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

def run_one(X, y, pipe_args, estimator):
    """
    X, y: features and target
    pipe_args: args to control FE pipeline
    estimator: the estmator
    
    Returns metrics, such as:
    val_scores: estimated val score for each each CV fold
    train_scores: estimated val score for each each CV fold
    """
    
    print("run_one: pipe_args: {}".format(pipe_args))
          
    steps = get_pipeline_steps(pipe_args)
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
