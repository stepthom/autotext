import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import uuid
import argparse
import numpy as np
import pandas as pd
import os

import json
import socket
import datetime


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

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--geo-id-set', type=int, default=1)
    parser.add_argument('-a', '--algo-set', type=int, default=1)
    
    args = parser.parse_args()
    runname = str(uuid.uuid4())
    
    id_col = 'building_id'
    target_col = 'damage_grade'
    out_dir = 'earthquake/out'
    data_id = '000'

    train_df  = pd.read_csv('earthquake/earthquake_train.csv')
    #train_df  = train_df.sample(frac=0.1, random_state=3)
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
        
    estimator_list = ['lgbm']
    ensemble = False
    if args.algo_set == 1:
        estimator_list = ['lgbm']
    elif args.algo_set == 2:
        estimator_list = ['xgboost']
    else:
        estimator_list = ['lgbm', 'xgboost']
        ensemble = True

    X = train_df.drop(['building_id', 'damage_grade'], axis=1)
    y = train_df['damage_grade']

    results = {}
    run_fn = os.path.join("earthquake/out", "tune_eq_{}.json".format(runname))
    print("tune_eq: Run name: {}".format(runname))

    results['runname'] = runname
    results['args'] = vars(args)
    results['hostname'] = socket.gethostname()
    results['starttime'] = str(datetime.datetime.now())

    automl_settings = {
        "time_budget": 20000,
        "log_file_name": "logs/flaml-{}.log".format(runname),
        "task": 'classification',
        "n_jobs": 8,
        "estimator_list": estimator_list,
        "model_history": False,
        "eval_method": "cv",
        "n_splits": 3,
        "metric": "micro_f1",
        "log_training_metric": True,
        "verbose": 1,
        "ensemble": ensemble,
    }
    clf = AutoML()
    clf.fit(X, y, **automl_settings)

    endtime = str(datetime.datetime.now())
    results['endtime'] = endtime
    results['automl_settings'] =  automl_settings
    results['best_score'] =  1 - clf.best_loss
    results['best_config'] =  clf.best_config
    


    print("Run name: {}".format(runname))
    print("Run file name: {}".format(run_fn))
   
    X_test = test_df.drop([id_col], axis=1)
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
