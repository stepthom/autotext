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

def main():    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-g', '--keep-top-id', type=int, default=1)
    parser.add_argument('-a', '--algo-set', type=int, default=1)
    
    args = parser.parse_args()
    runname = str(uuid.uuid4())
    
    id_col = 'respondent_id'
    target_col = 'h1n1_vaccine'
    out_dir = 'h1n1/out'
    data_id = '000'
    metric = "roc_auc"
    
    train_fn = "h1n1/data/vaccine_h1n1_train_481e85dd-2af9-4abc-b5f6-fb0f66a94ed3.csv"
    test_fn = "h1n1/data/vaccine_h1n1_test_481e85dd-2af9-4abc-b5f6-fb0f66a94ed3.csv"

    train_df  = pd.read_csv(train_fn)
    #train_df  = train_df.sample(frac=0.1, random_state=3)
    test_df  = pd.read_csv(test_fn)
    
        
    estimator_list = ['lgbm']
    ensemble = False
    if args.algo_set == 1:
        estimator_list = ['lgbm']
    elif args.algo_set == 2:
        estimator_list = ['xgboost']
    elif args.algo_set == 3:
        estimator_list = ['lgbm', 'xgboost', 'catboost']
        ensemble = True
    elif args.algo_set == 4:
        estimator_list = ['catboost']

    X = train_df.drop([id_col, target_col], axis=1)
    y = train_df[target_col]

    results = {}
    run_fn = os.path.join(out_dir, "tune_{}.json".format(runname))
    print("tune: Run name: {}".format(runname))

    results['runname'] = runname
    results['args'] = vars(args)
    os_steve_min_sample_leaf =  int(os.environ.get('OS_STEVE_MIN_SAMPLE_LEAF', "10"))
    os_steve_smoothing       =  float(os.environ.get('OS_STEVE_SMOOTHING', "0.1"))
    results['os_steve_min_sample_leaf'] = os_steve_min_sample_leaf
    results['os_steve_smoothing'] = os_steve_smoothing
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
    print("tune: Wrote preds file: {}".format(preds_fn))

    probas = clf.predict_proba(X_test)
    columns = None
    if hasattr(clf, 'classes_'):
        columns = clf.classes_
    probas_df = pd.DataFrame(probas, columns=columns)
    probas_df[id_col] = test_df[id_col]
    probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
    probas_fn = os.path.join(out_dir, "{}-{}-probas.csv".format(runname, data_id))
    probas_df.to_csv(probas_fn, index=False)
    print("tune: Wrote probas file: {}".format(probas_fn))

    results['preds_fn'] = preds_fn
    results['probas_fn'] = probas_fn
    dump_json(run_fn, results)

if __name__ == "__main__":
    main()
