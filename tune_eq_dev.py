import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
# import comet_ml at the top of your file
import uuid
import argparse
import numpy as np
import pandas as pd
import os
import time

from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import scipy.stats

import optuna

from SteveHelpers import dump_json, get_data_types, read_json
from SteveHelpers import check_dataframe, check_array
from SteveHelpers import estimate_metrics
from SteveHelpers import StudyData
from SteveHelpers import SteveEncoder, SteveNumericCapper, SteveFeatureCombinerEQ, SteveNumericNormalizer

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    parser.add_argument('--sample-frac', type=float, default=1.0)
    
    # Optuna params
    parser.add_argument('--n-trials', type=int, default=500)

    args = parser.parse_args()
    args = vars(args)
    
    study_name = "eq"
    
    print("Command line arguments:")
    print(args)
        
    def objective(trial, study_data):
        
        cat_cols = [
                "geo_level_1_id", "geo_level_2_id", "geo_level_3_id",
                "land_surface_condition", "foundation_type", "roof_type", 
                "ground_floor_type", "other_floor_type", "position", "plan_configuration",  "legal_ownership_status"]
        
        steps = []
        steps.append(('ord_encoder',  SteveEncoder( 
            cols=cat_cols, suffix="_oenc", drop_orig=True,
            encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32),
        )))
        steps.append(('combiner1', SteveFeatureCombinerEQ()))
        steps.append(('num_capper', SteveNumericCapper(num_cols=['age'], max_val=30)))
                     
        num_cols = [ "count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families",
                   "height_mult_area", "families_div_floors", "area_mult_age", "area_div_floors"]
        steps.append(('num_normalizer', SteveNumericNormalizer(num_cols, drop_orig=True)))
        
        pipe = Pipeline(steps)
    
        upper = 4096
        params = {
              "max_bin": trial.suggest_int("max_bin", 32, 512),
              "num_leaves": trial.suggest_int("num_leaves", 4, upper, log=True),
              "min_child_weight": trial.suggest_loguniform("min_child_weight", 0.001, 128),
              "min_child_samples": trial.suggest_int("min_child_samples", 2, 2**7, log=True),
              "learning_rate": trial.suggest_loguniform("learning_rate", 1/256, 2.0),
              "feature_fraction_bynode":  trial.suggest_float("feature_fraction_bynode", 0.01, 1.0),
              "colsample_bytree": trial.suggest_float("colsample_bytree", 0.01, 1.0),
              "subsample_freq": 10,
              "subsample": trial.suggest_float("subsample", 0.10, 1.0),
              "reg_alpha": trial.suggest_loguniform("reg_alpha", 1/1024, 1024),
              "reg_lambda": trial.suggest_loguniform("reg_lambda", 1/1024, 1024),
            
              "cat_smooth": trial.suggest_float("cat_smooth", 5, 25),
              "cat_l2": trial.suggest_float("cat_l2", 5, 25),
              "min_data_per_group": trial.suggest_int("min_data_per_group", 50, 200),
            
              "extra_trees": False,
              "is_unbalance": True,
              "n_estimators": 3000,
              "boosting_type": 'gbdt',
              "max_depth": -1,
              "n_jobs": 5,
              "verbosity": -1,
              "seed": 77,
        }
        estimator = LGBMClassifier(**params)
        print("estimator: {}".format(estimator))
            
        metrics = {}
        val_scores = []
        train_scores = []
        train_times = []
        best_iterations = []

        num_cv = 6
        skf = StratifiedKFold(n_splits=num_cv, random_state=42, shuffle=True)

        cv_step = -1
        for train_index, val_index in skf.split(study_data.X, study_data.y):
            cv_step = cv_step + 1
            print("========================================")
            print("estimate_metrics: cv_step {} of {}".format(cv_step, num_cv))

            X_train, X_val = study_data.X.loc[train_index].reset_index(drop=True), study_data.X.loc[val_index].reset_index(drop=True)
            y_train, y_val = study_data.y[train_index], study_data.y[val_index]

            pipe.fit(X_train, y_train)

            _X_train = pipe.transform(X_train)
            _X_val  = pipe.transform(X_val)
            #_X_train.to_csv("tmpXTrain.csv")
            #_X_val.to_csv("tmpXVal.csv")

            #check_dataframe(_X_train, "_X_train", full=False)
            #check_dataframe(_X_val, "_X_val", full=False)

            indices =  [i for i, ix in enumerate(_X_train.columns.values) if "_oenc" in ix]
            print("categorical indices: {}".format(indices))
            extra_fit_params = {
                    'eval_set': [(_X_val, y_val)],
                    'early_stopping_rounds': 50,
                    'verbose': 200,
                    'categorical_feature': indices, 
                }

            start = time.time()
            print("estimate_metric: fitting...: ")
            estimator.fit(_X_train, y_train, **extra_fit_params)
            train_time = time.time() - start
            print("...done {} secs: ".format(train_time))

            print("estimate_metric: calc Val metrics. ")
            #y_val_pred_proba = estimator.predict_proba(_X_val)
            y_val_pred = estimator.predict(_X_val)
            val_score = f1_score(y_val, y_val_pred, average="micro")
            print("val_score = {}".format(val_score))
            #print(classification_report(y_val, y_val_pred, digits=4))
            

            print("estimate_metric: calc Train metrics. ")
            #y_train_pred_proba = estimator.predict_proba(_X_train)
            y_train_pred = estimator.predict(_X_train)
            train_score =  f1_score(y_train, y_train_pred, average="micro")
            #print(classification_report(y_train, y_train_pred, digits=4))

            bi = None
            if hasattr(estimator, 'best_iteration_'):
                bi = estimator.best_iteration_
            elif hasattr(estimator, 'best_iteration'):
                bi = estimator.best_iteration
            print("best iteration: {}".format(bi))
            
            train_times.append(train_time)
            val_scores.append(val_score)
            train_scores.append(train_score)
            best_iterations.append(bi)

        def mean_confidence_interval(data, confidence=0.95):
            a = 1.0 * np.array(data)
            n = len(a)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
            return m, m-h, m+h
       
        metrics = {}
        metrics['train_times'] = train_times
        metrics['val_scores'] = val_scores
        metrics['val_scores_range'] = mean_confidence_interval(val_scores)
        metrics['train_scores'] = train_scores
        metrics['best_iterations'] = best_iterations
        bir = []
        if len(best_iterations) > 0:
            bir = mean_confidence_interval(best_iterations)
        metrics['best_iterations_range'] = bir

        print("CV complete. metrics:")
        print(metrics)

        # Log for later
        trial.set_user_attr("estimator_params", params)
        trial.set_user_attr("metrics", metrics)

        return np.mean(val_scores)
    
    study = optuna.load_study(study_name=study_name, storage=args['storage'])
    study_data = StudyData(study, args['sample_frac'])
    
    study.optimize(lambda trial: objective(trial, study_data),
                    n_trials=args['n_trials'], 
                    gc_after_trial=True)
    return
    
if __name__ == "__main__":
    main()
