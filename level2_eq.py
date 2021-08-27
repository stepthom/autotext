import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import argparse
import numpy as np
import pandas as pd
import os

import json
from json.decoder import JSONDecodeError

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import SteveHelpers
          
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--project', type=str, default="eq")
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--skip-output', type=int, default=0)
    args = parser.parse_args()
    args = vars(args)

    if args['project'] == "eq":
        from EQHelpers import get_project, get_pipeline_steps
        trial_logs = [

            # lgbm, 
            #"earthquake/out/-trial-.json",

            # lgbm, 0.7553, 0.7533
            "earthquake/out/583b7931b2274e45839372114f3dc80f-trial-43.json",

            # lgbm, 0.7544, 0.7519
            #"earthquake/out/583b7931b2274e45839372114f3dc80f-trial-21.json",

            # lgbm, 0.7548, 
            #"earthquake/out/583b7931b2274e45839372114f3dc80f-trial-22.json",

            # xgboost, 0.7516, 0.7437
            #"earthquake/out/b7bce481fcab40de88418a34fc05f4c0-trial-26.json",

            # hist, 0.7480, 0.7458
            #"earthquake/out/0a7930f91e194ed2b0d18b9a01352b31-trial-67.json", 
        ]
    elif args['project'] == "h1n1":
        from H1N1Helpers import get_project, get_pipeline_steps
        trial_logs = []
        
    p = get_project(args['sample_frac'])
    
    
    i = 0
    for trial_log in trial_logs:
        i = i + 1
        run = SteveHelpers.read_json(trial_log)
       
        run_args = run['args']
        pipe_args = run['pipe_args']
        params = run['params']
        metrics = run['metrics']
        runname = run['runname']
        trial = run['trial']
        
        n_estimators = None
        bi = metrics.get('best_iterations_range', [])
        if len(bi) > 0:
            
            # This seems to be about best
            n_estimators = np.floor(max(bi)).astype(int)
            
        estimator = None
        if run_args['estimator_name'] == "lgbm":
            params['n_estimators'] = n_estimators
            estimator = LGBMClassifier(**params)
        elif run_args['estimator_name'] == "xgboost":
            params['n_estimators'] = n_estimators
            estimator = XGBClassifier(**params)
        elif run_args['estimator_name'] == "rf":
            estimator = RandomForestClassifier(**params)
        elif run_args['estimator_name'] == "lr":
            estimator = LogisticRegression(**params)
        elif run_args['estimator_name'] == "hist":
            estimator = HistGradientBoostingClassifier(**params)
        else:
            print("Unknown estimator name {}".format(estimator_name))
                           
        pipe, estimator = SteveHelpers.run_one(p.X, p.y, pipe_args, get_pipeline_steps, estimator)
        
        if args["skip_output"] == 1:
            print("Not producing predictions files. Returning.")
            return
                                             
        _X_test = pipe.transform(p.X_test)
        probas = estimator.predict_proba(_X_test)
        preds  = estimator.predict(_X_test)
        preds = p.label_transformer.inverse_transform(preds)

        preds_df = pd.DataFrame(data={'id': p.test_df[p.id_col], p.target_col: preds})
        preds_fn = os.path.join(p.out_dir, "{}-{}-preds.csv".format(runname, trial))
        preds_df.to_csv(preds_fn, index=False)
        print("level2: Wrote preds file: {}".format(preds_fn))

        probas_df = pd.DataFrame(probas, columns=p.label_transformer.classes_)
        probas_df[p.id_col] = p.test_df[p.id_col]
        probas_df = probas_df[ [p.id_col] + [ col for col in probas_df.columns if col != p.id_col ] ]
        probas_fn = os.path.join(p.out_dir, "{}-{}-probas.csv".format(runname, trial))
        probas_df.to_csv(probas_fn, index=False)
        print("level2: Wrote probas file: {}".format(probas_fn))

    
if __name__ == "__main__":
    main()
