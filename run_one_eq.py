import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import uuid
import argparse
import numpy as np
import pandas as pd
import os
import sys

import scipy.stats
import json
from json.decoder import JSONDecodeError
import socket
import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgbm
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import category_encoders as ce
from category_encoders.wrapper import PolynomialWrapper
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import StackingClassifier
#from mlxtend import StackingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from SteveHelpers import dump_json, get_data_types
from EQHelpers import get_pipeline_steps, run_one
from EQHelpers import get_pipeline_steps, estimate_metrics

          
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--id-col', type=str, default="building_id")
    parser.add_argument('--target-col', type=str, default="damage_grade")
    parser.add_argument('--out-dir', type=str, default="earthquake/out")
    
    # Prep/FE params
    parser.add_argument('--sample-frac', type=float, default=1.0)
    
    args = parser.parse_args()
    args = vars(args)

    print("Command line arguments:")
    print(args)

    data_id = '000'

    train_fn = 'earthquake/earthquake_train.csv'
    test_fn = 'earthquake/earthquake_test.csv'
    train_df = pd.read_csv(train_fn)
    test_df = pd.read_csv(test_fn)
    if args['sample_frac']< 1.0:
        train_df  = train_df.sample(frac=args['sample_frac'], random_state=3).reset_index(drop=True)
        
    X = train_df.drop([args['id_col'], args['target_col']], axis=1)
    X_test = test_df.drop([args['id_col']], axis=1)
    y = train_df[args['target_col']]
    label_transformer = LabelEncoder()
    y = label_transformer.fit_transform(y)
    
    pipe_args = {
        "drop_cols": [],
        "cat_cols_onehot_encode": [],
        "cat_cols_target_encode": [],
        "cat_cols_ordinal_encode": ["land_surface_condition", "foundation_type", "roof_type", "ground_floor_type", "other_floor_type", "position", "plan_configuration", "legal_ownership_status", "geo_level_1_id", "geo_level_2_id", "geo_level_3_id"], 
        "float_cols": ["count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families"],
        "autofeat": 0,
        "normalize": 0
    }
    params = {
        "n_estimators": 2000,
        "num_leaves": 19,
        "min_child_weight": 0.1612338676485728,
        "min_child_samples": 81,
        "learning_rate": 0.27055394555587947,
        "feature_fraction_bynode": 0.47347927279418395,
        "colsample_bytree": 0.518249963175708,
        "subsample": 1.0,
        "subsample_freq": 0,
        "reg_alpha": 0.00696053068769837,
        "reg_lambda": 472.19095099171886,
        "extra_trees": False,
        "n_estimators": 3500,
        "boosting_type": "gbdt",
        "max_depth": -1,
        "n_jobs": 5,
        "objective": "multiclass",
        "verbosity": -1,
        "num_class": 3,
        "seed": 77,
    }
        
    estimator = LGBMClassifier(**params)
    metrics = estimate_metrics(X, y, pipe_args, estimator, num_cv=12, early_stop=True)
    print(metrics)
if __name__ == "__main__":
    main()
