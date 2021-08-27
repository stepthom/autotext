import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np

#import time

#from xgboost import XGBClassifier
#from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
#from sklearn.metrics import classification_report, f1_score
#from sklearn.model_selection import StratifiedKFold
#from sklearn.experimental import enable_hist_gradient_boosting  # noqa
#from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import scipy.stats

import category_encoders as ce

from SteveHelpers import *

def get_project(sample_frac=1.0):
    p = Project(
        id_col ="building_id",
        target_col = "damage_grade",
        out_dir = "earthquake/out",
        train_fn = 'earthquake/earthquake_train.csv',
        test_fn = 'earthquake/earthquake_test.csv',
        objective = 'multiclass',
        num_class = 3,
        metric="f1",
        sample_frac = sample_frac,
    )

    return p

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
        
    return steps


