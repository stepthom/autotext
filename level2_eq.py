import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import argparse
import numpy as np
import pandas as pd
import os
from pprint import pprint; 

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import optuna

from SteveHelpers import StudyData, run_one
          
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--skip-output', type=int, default=0)
    args = parser.parse_args()
    
    trials_to_run = {}
    trials_to_run["h1n1"]  = [3]
    trials_to_run["seasonal"]  = [0, 1]
    
    for study_name, trial_numbers in trials_to_run.items():
        print("Study name {}".format(study_name))

        study = optuna.load_study(study_name=study_name, storage=args.storage)
        study_data = StudyData(study, args.sample_frac)

        for trial_number in trial_numbers:
            print("Trial number {}".format(trial_number))
            trial =  study.trials[trial_number]
            params = trial.params
            estimator_name = trial.user_attrs['estimator_name']
            estimator_params = trial.user_attrs['estimator_params']
            metrics = trial.user_attrs['metrics']

            pprint(params)
            pprint(metrics)
            pprint(trial.user_attrs['proba_fn'])
            
            n_estimators = None
            bi = metrics.get('best_iterations_range', [])
            if len(bi) > 0:
                # This seems to be about best
                n_estimators = np.floor(max(bi)).astype(int)

            estimator = None
            if estimator_name == "lgbm":
                params['n_estimators'] = n_estimators
                estimator = LGBMClassifier(**estimator_params)
            elif estimator_name == "xgboost":
                params['n_estimators'] = n_estimators
                estimator = XGBClassifier(**estimator_params)
            elif estimator_name == "rf":
                estimator = RandomForestClassifier(**estimator_params)
            elif estimator_name == "lr":
                estimator = LogisticRegression(**estimator_params)
            elif estimator_name == "hist":
                estimator = HistGradientBoostingClassifier(**estimator_params)
            else:
                print("Unknown estimator name {}".format(estimator_name))

            pipe, estimator = run_one(study_data, study_name, params['pipe_name'], estimator)

            if args.skip_output == 1:
                print("Not producing predictions files. Returning.")
                return

            _X_test = pipe.transform(study_data.X_test)
            probas = estimator.predict_proba(_X_test)
            preds  = estimator.predict(_X_test)
            preds = study_data.label_transformer.inverse_transform(preds)

            id_col = study.user_attrs['id_col']
            target_col = study.user_attrs['target_col']
            out_dir = study.user_attrs['out_dir']
            
            preds_df = pd.DataFrame(data={'id': study_data.test_df[id_col], target_col: preds})
            preds_fn = os.path.join(out_dir, "{}-{}-preds.csv".format(study_name, trial_number))
            preds_df.to_csv(preds_fn, index=False)
            print("level2: Wrote preds file: {}".format(preds_fn))

            probas_df = pd.DataFrame(probas, columns=study_data.label_transformer.classes_)
            probas_df[id_col] = study_data.test_df[id_col]
            probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
            probas_fn = os.path.join(out_dir, "{}-{}-probas.csv".format(study_name, trial_number))
            probas_df.to_csv(probas_fn, index=False)
            print("level2: Wrote probas file: {}".format(probas_fn))
            
if __name__ == "__main__":
    main()
