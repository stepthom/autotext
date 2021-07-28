import uuid
import argparse
import numpy as np
import pandas as pd
import os

import json
import socket
import datetime


from random import shuffle

#from sklearn.model_selection import cross_val_score, train_test_split
#from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
#from sklearn.model_selection import ShuffleSplit
#import lightgbm as lgb
#from sklearn.metrics import classification_report
#from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
#from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, RandomForestRegressor
#from sklearn.linear_model import LogisticRegression

from scipy.stats import uniform, randint


import jsonpickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


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
    parser.add_argument('-d', '--data-sheet', default="random")
    parser.add_argument('-c', '--competition', default='pump')
    parser.add_argument('-s', '--run-settings', default='default_settings.json')
    
    args = parser.parse_args()
    
    with open(args.run_settings) as f:
        run_settings = json.load(f)
       
    # Get the struct that holds the settings for this comp
    comp_settings = run_settings.get(args.competition, None)
    if comp_settings is None:
        print('run_one: Error. Invalid run settings.')
        return
    
    target_col = comp_settings.get('target_col', '')
    id_col = comp_settings.get('id_col', '')
    data_dir = comp_settings.get('data_dir', '')
    out_dir = comp_settings.get('out_dir', '')
    eval_metric = comp_settings.get('eval_metric', '')
    search_time = comp_settings.get('search_time', 200)
    search_type = comp_settings.get('search_type', 'FLAML')
    ensemble = comp_settings.get('ensemble', True)
   
    # Helper function to see if a datasheet should be run. i.e.
    # has not already been run with same run settings
    def should_run(data_sheet_file, runs, comp_settings):
        data_sheet = {}
        with open(data_sheet_file) as f:
            data_sheet = json.load(f)
        
        # Tmp hack because of FLAML bug #130
        cat_enc = data_sheet.get('cat_encoder', '')
        if cat_enc == "None":
            print("run_one: Skipping this data sheet because cat_encoder is None and FLAML bug #130. Skipping.")
            return False
        
        for run in runs:
            if (run.get('data_id', '') == data_sheet.get('data_id', '') and
                run.get('comp_settings', '') == comp_settings):  
                print("run_one: Run already completed for data sheet and settings. Skipping.")
                print("run_one: (Run was {}.)".format(run.get('runname'), ''))
                return False
           
        return True
    
    
    # Parse the out_dir, find all existing runs
    runs = []
    for file in os.listdir(out_dir): 
        if file.startswith("run") and file.endswith(".json"):
            with open(os.path.join(out_dir, file)) as f:
                run = json.load(f)
                runs.append(run)
    print("run_one: Found {} existing runs of all kinds.".format(len(runs)))
           
   
    # Get list of potential data sheets to run
    data_sheet_files = []
    if args.data_sheet == "random":
        # Parse the data_dir, find all data sheets
        for file in os.listdir(data_dir):
            if file.endswith(".json"):
                data_sheet_files.append(os.path.join(data_dir, file))
                                             
        print("run_one: Found {} potential data sheets.".format(len(data_sheet_files)))
        shuffle(data_sheet_files)
    else:
        data_sheet_files.append(args.data_sheet)
    
    
    # Find a datasheet to run!
    data_sheet_to_run = None
    for data_sheet_file in data_sheet_files:
        print("run_one: Considering data sheet {}".format(data_sheet_file))
        if should_run(data_sheet_file, runs, comp_settings):
            data_sheet_to_run = data_sheet_file
            break
        else:
            print("run_one: Shouldn't run {}".format(data_sheet_file))

    if data_sheet_to_run is None:
        print("run_one: Couldn't find any data sheets to run.")
        return
        
    data_sheet = {}
    with open(data_sheet_to_run) as f:
        data_sheet = json.load(f)
        
    data_id = data_sheet.get('data_id', None)
    config_summary = data_sheet.get('config_summary', None)
    if data_id is None or config_summary is None:
        raise Exception("Error: Data sheet does not contain data id or config summary.")

    runname = str(uuid.uuid4())
    print("run_one: Running data_id {}".format(data_id))
    print("run_one: Running config_summary {}".format(config_summary))
    print("run_one: Run name: {}".format(runname))

    train_fn = data_sheet.get('fn_train', None)
    test_fn = data_sheet.get('fn_test', None)

    if train_fn is None or test_fn is None:
        raise Exception("Error: cannot find train or test file names")

    train_fn = os.path.join(data_dir, os.path.basename(train_fn))
    test_fn = os.path.join(data_dir, os.path.basename(test_fn))

    print("DEBUG: Reading training data {}".format(train_fn))
    train_df = pd.read_csv(train_fn)

    X_train = train_df.drop([target_col, id_col], axis=1)
    y_train = train_df[target_col]

    print("DEBUG: Reading testing data {}".format(test_fn))
    test_df = pd.read_csv(test_fn)
    X_test = test_df.drop([id_col], axis=1)
    
    pipe = None
    automl_settings = {}
    starttime = str(datetime.datetime.now())
    val_score = 0
    if search_type == "FLAML":
        from flaml import AutoML
        automl_settings = {
            "time_budget": search_time,
            "task": 'classification',
            "log_file_name": "logs/flaml-{}.log".format(runname),
            "n_jobs": 8,
            "estimator_list": ['lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost'],
            "model_history": False,
            "eval_method": "cv",
            "n_splits": 3,
            "metric": eval_metric,
            "log_training_metric": True,
            "verbose": 1,
            "ensemble": ensemble,
        }
        pipe = AutoML()
        pipe.fit(X_train, y_train, **automl_settings)
        val_score = 1-pipe.best_loss
        
    elif search_type == "autogluon":
        from autogluon.tabular import TabularDataset, TabularPredictor
        
        _df = X_train.copy()
        _df[target] = y_train
        
        excluded_model_types = ['KNN', 'NN']
        pipe = TabularPredictor(label=target, eval_metric=eval_metric)

        pipe.fit(_df, time_limit=search_time, presets="best_quality", excluded_model_types=excluded_model_types)
       
        leaderboard_df = pipe.leaderboard()
        print(leaderboard_df.head())
        val_score = leaderboard.head(1)['score_val'].item()
        
    elif search_type == "mljar":
        from supervised.automl import AutoML # mljar-supervised
        
        automl_settings = {
            "mode": "Compete",
            "model_time_limit": search_time,
            "eval_metric": eval_metric,
            "stack_models": True,
            "explain_level": 0,
            "features_selection": False,
            "algorithms": ["Linear", "Decision Tree", "Random Forest", "LightGBM", "Xgboost", "CatBoost", "Neural Network"],
            "golden_features": True,
            "results_path": "logs/{}".format(runname),
            "n_jobs": 8,
            "verbose": 1,
        }
        
        pipe = AutoML(**automl_settings)
        pipe.fit(X_train, y_train)
        
        # TODO: get val_score somehow
        
    
    endtime = str(datetime.datetime.now())
    
    print("run_one: val score: {}".format(val_score))
   
    preds = pipe.predict(X_test)
    preds_df = pd.DataFrame(data={'id': test_df[id_col], target_col: preds})
    preds_fn = os.path.join(out_dir, "{}-{}-preds.csv".format(runname, data_id))
    preds_df.to_csv(preds_fn, index=False)
    print("run_one: Wrote preds file: {}".format(preds_fn))

    probas = pipe.predict_proba(X_test)
    columns = None
    if hasattr(pipe, 'classes_'):
        columns = pipe.classes_
    probas_df = pd.DataFrame(probas, columns=columns)
    probas_df[id_col] = test_df[id_col]
    probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
    probas_fn = os.path.join(out_dir, "{}-{}-probas.csv".format(runname, data_id))
    probas_df.to_csv(probas_fn, index=False)
    print("run_one: Wrote probas file: {}".format(probas_fn))

    # This structure will hold all the results and will be dumped to disk.
    run = {}
    run['runname'] = runname
    run['hostname'] = socket.gethostname()
    run['competition'] = args.competition
    run['comp_settings'] = comp_settings
    run['search_time'] = search_time
    run['search_type'] = search_type
    run['eval_metric'] = eval_metric
    run['data_id'] = data_id
    run['data_sheet'] = data_sheet

    run['automl_settings'] = jsonpickle.encode(automl_settings, unpicklable=False, keys=True)
    run['starttime'] = starttime
    run['endtime'] = endtime
    run['best_estimator'] = pipe.best_estimator
    run['best_config'] = pipe.best_config
    run['best_model'] = '{}'.format(str(pipe.model))
    run['val_score'] = val_score

    run['preds_fn'] = preds_fn
    run['probas_fn'] = probas_fn

    run_fn = os.path.join(out_dir, "run_{}.json".format(runname))
    dump_json(run_fn, run)

    print("run_one: Run name: {}".format(runname))
    print("run_one: Run file name: {}".format(run_fn))
    print("run_one: Config summary: {}".format(config_summary))

if __name__ == "__main__":
    main()
