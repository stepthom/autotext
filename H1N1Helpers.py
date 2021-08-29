import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

import time

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder 
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import scipy.stats

import category_encoders as ce

from SteveHelpers import *

def get_project(sample_frac=1.0):
    p = Project(
        id_col = "respondent_id",
        target_col = "h1n1_vaccine",
        out_dir = "h1n1/out",
        train_fn = "h1n1/vaccine_h1n1_train.csv",
        test_fn = "h1n1/vaccine_h1n1_test.csv",

        objective = 'binary',
        num_class = 1,
        metric="roc_auc",
        sample_frac = sample_frac,
    )

    return p

def get_pipe_args(pipe_args_name):
    pipe_args = {}
    
    if pipe_args_name == "h1n1_01":
        
    return pipe_args

        

