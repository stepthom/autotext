import uuid
import sys
import argparse
import numpy as np
import pandas as pd
import os

import json
import socket
import datetime

import gc

from random import shuffle

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
#import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from scipy.stats import uniform, randint

#import ConfigSpace.read_and_write.json as config_json

import jsonpickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

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
        
    print("DEBUG: Writing json file {}".format(fn))
    with open(fn, 'w') as fp:
        json.dump(json_obj, fp, indent=4, cls=NumpyEncoder)


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-sheet', default=None)
    parser.add_argument('-c', '--competition', default='pump')
    
    args = parser.parse_args()
    
    if args.competition == "pump":
        target_col = "status_group"
        id_col = "id"
        data_dir = "pump/data"
        out_dir = "pump/out"
        eval_metric = "accuracy"
    elif args.competition == "h1n1":
        target_col = "h1n1_vaccine"
        id_col = "respondent_id"
        data_dir = "h1n1/data"
        out_dir = "h1n1/out"
        eval_metric = "roc_auc"
    elif args.competition == "seasonal":
        target_col = "seasonal_vaccine"
        id_col = "respondent_id"
        data_dir = "seasonal/data"
        out_dir = "seasonal/out"
        eval_metric = "roc_auc"
    elif args.competition == "earthquake":
        target_col = "damage_grade"
        id_col = "building_id"
        data_dir = "earthquake/data"
        out_dir = "earthquake/out"
        eval_metric = "micro_f1"
    else:
        print("Error: unknown competition type: {}".format(args.competition))
        return
    
    data_sheet_files = []
    if args.data_sheet is not None:
        data_sheet_files = [args.data_sheet]
        
    else:
        # Read directory
        for file in os.listdir(data_dir):
            if file.endswith(".json"):
                data_sheet_files.append(os.path.join(data_dir, file))
        
        shuffle(data_sheet_files)
        print("DEBUG: Found {} data sheets.".format(len(data_sheet_files)))
    
        
    def run_data_sheet(data_sheet, target_col, id_col, data_dir, out_dir, eval_metric): 
        
        data_id = data_sheet.get('data_id', None)
        config_summary = data_sheet.get('config_summary', None)
        if data_id is None or config_summary is None:
            raise Exception("Error: Data sheet does not contain data id or config summary.")
            
        print("DEBUG: Running data_id {}".format(data_id))
        print("DEBUG: Running config_summary {}".format(config_summary))
        
        search_time = 1000
        search_type = "FLAML"
        
        # Check if we even need to run this
        runs = data_sheet.get('runs', {})
        for run in runs:
            if (runs[run].get('search_type', '') == search_type and 
                runs[run].get('search_time', 0) == search_time and
                runs[run].get('eval_metric', '') == eval_metric):
                print("Early stopping. Run already completed for data sheet. Skipping.")
                return data_sheet
            
        # Tmp hack because of FLAML bug #130
        cat_enc = data_sheet.get('cat_encoder', '')
        if cat_enc == "None":
            print("Skipping this data sheet because cat_encoder is None and FLAML bug #130. Skipping.")
            return data_sheet
        
        # This structure will hold all the results and will be dumped to disk.
        run = {}
        runname = str(uuid.uuid4())
        run['runname'] = runname
        run['hostname'] = socket.gethostname()
        run['search_type'] = search_type
        run['search_time'] = search_time
        run['eval_metric'] = eval_metric
        
        print("Run name: {}".format(runname))
        
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
    
        pipe = AutoML()
        automl_settings = {
            "time_budget": search_time,
            "task": 'classification',
            "log_file_name": "{}/flaml-{}.log".format(out_dir, runname),
            "n_jobs": 5,
            "estimator_list": ['lgbm', 'xgboost', 'rf', 'extra_tree', 'catboost'],
            "model_history": True,
            "eval_method": "cv",
            "n_splits": 3,
            "metric": eval_metric,
            "log_training_metric": True,
            "verbose": 1,
            "ensemble": True,
        }

        run['flaml_settings'] = jsonpickle.encode(automl_settings, unpicklable=False, keys=True)

        run['starttime'] = str(datetime.datetime.now())
        pipe.fit(X_train, y_train, **automl_settings)
        run['endtime'] = str(datetime.datetime.now())

        run['best_estimator'] = pipe.best_estimator
        run['best_config'] = pipe.best_config
        run['best_model'] = '{}'.format(str(pipe.model))
        run['val_score'] = 1-pipe.best_loss
        
        print("FLAML val score: {}".format(run['val_score']))
    
        preds = pipe.predict(X_test)
        preds_df = pd.DataFrame(data={'id': test_df[id_col], target_col: preds})
        preds_fn = os.path.join(out_dir, "{}-{}-preds.csv".format(data_id, runname))
        preds_df.to_csv(preds_fn, index=False)
        
        probas = pipe.predict_proba(X_test)
        columns = None
        if hasattr(pipe, 'classes_'):
            columns = pipe.classes_
        probas_df = pd.DataFrame(probas, columns=columns)
        probas_df[id_col] = test_df[id_col]
        probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
        probas_fn = os.path.join(out_dir, "{}-{}-probas.csv".format(data_id, runname))
        probas_df.to_csv(probas_fn, index=False)
        
        run['endtime'] = str(datetime.datetime.now())
        
        run['preds_fn'] = preds_fn
        run['probas_fn'] = probas_fn
        
        runs[runname] = run
        
        data_sheet['runs'] = runs
    
        print("DEBUG: Run name: {}".format(runname))
        print("DEBUG: Config summary: {}".format(config_summary))
        
        del pipe
        gc.collect()
        
        return data_sheet
    
    
    for data_sheet_file in data_sheet_files:
        data_sheet = {}
        with open(data_sheet_file) as f:
            data_sheet = json.load(f)
            
        data_sheet = run_data_sheet(data_sheet, target_col, id_col, data_dir, out_dir, eval_metric)
        dump_json(data_sheet_file, data_sheet)

if __name__ == "__main__":
    main()
