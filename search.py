from flaml import AutoML
import uuid
import sys
import argparse
import numpy as np

import json

import pandas as pd
import socket

from sklearn.model_selection import train_test_split
import datetime

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import uniform, randint

import ConfigSpace.read_and_write.json as config_json

import jsonpickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_metrics(y_true, y_pred):
    res = {}
    res['accuracy'] = accuracy_score(y_true, y_pred)
    res['f1'] = f1_score(y_true, y_pred, average="macro")
    res['recall'] = recall_score(y_true, y_pred, average="macro")
    res['precision'] = precision_score(y_true, y_pred, average="macro")
    res['report'] = classification_report(y_true, y_pred, output_dict=True)
    return res

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dump_results(runname, results):

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

    with open('out/{}-results.json'.format(runname), 'w') as fp:
        json.dump(results, fp, indent=4, cls=NumpyEncoder)


# Helper function to print out the results of hyperparmater tuning in a nice table.
def cv_results_to_df(cv_results):
    results = pd.DataFrame(list(cv_results['params']))
    results['mean_fit_time'] = cv_results['mean_fit_time']
    results['mean_score_time'] = cv_results['mean_score_time']
    results['mean_train_score'] = cv_results['mean_train_score']
    results['std_train_score'] = cv_results['std_train_score']
    results['mean_test_score'] = cv_results['mean_test_score']
    results['std_test_score'] = cv_results['std_test_score']
    results['rank_test_score'] = cv_results['rank_test_score']

    results = results.sort_values(['mean_test_score'], ascending=False)
    return results


def custom_metric(X_test, y_test, estimator, labels, X_train, y_train,
                  weight_test=None, weight_train=None):
    from sklearn.metrics import accuracy_score
    y_pred = estimator.predict(X_test)
    test_loss = 1.0-accuracy_score(y_test, y_pred, sample_weight=weight_test)
    y_pred = estimator.predict(X_train)
    train_loss = 1.0-accuracy_score(y_train, y_pred, sample_weight=weight_train)
    alpha = 1.1
    print(test_loss * (1 + alpha) - alpha * train_loss, [test_loss, train_loss])
    return test_loss * (1 + alpha) - alpha * train_loss, [test_loss, train_loss]


def custom_metric1(X_test, y_test, estimator, labels, X_train, y_train,
                  weight_test=None, weight_train=None):
    from sklearn.metrics import f1_score
    y_pred = estimator.predict(X_test)
    test_loss = 1.0-f1_score(y_test, y_pred, labels=labels,
                         sample_weight=weight_test, average='macro')
    y_pred = estimator.predict(X_train)
    train_loss = 1.0-f1_score(y_train, y_pred, labels=labels,
                          sample_weight=weight_train, average='macro')
    alpha = 0.1
    return test_loss * (1 + alpha) - alpha * train_loss, [test_loss, train_loss]

def custom_metric2(X_test, y_test, estimator, labels, X_train, y_train,
                  weight_test=None, weight_train=None):
    from sklearn.metrics import log_loss
    y_pred = estimator.predict_proba(X_test)
    test_loss = log_loss(y_test, y_pred, labels=labels,
                         sample_weight=weight_test)
    y_pred = estimator.predict_proba(X_train)
    train_loss = log_loss(y_train, y_pred, labels=labels,
                          sample_weight=weight_train)
    alpha = 0.5
    return test_loss * (1 + alpha) - alpha * train_loss, [test_loss, train_loss]


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "settings_file", help="Path to JSON settings/config file.")
    args = parser.parse_args()
        
    runname = str(uuid.uuid4())

    print("Run name: {}".format(runname))

    # This structure will hold all the results and will be dumped to disk.
    results = {}
    results['runname'] = runname
    results['starttime'] = str(datetime.datetime.now())
    results['hostname'] = socket.gethostname()
    
    # Read the settings file
    with open(args.settings_file) as f:
        config = json.load(f)

    results['settings'] = config
    
    train_fn = config.get('train_filename', None)
    test_fn = config.get('test_filename', None)
    target = config.get('target', 'status_group')
    id_col = config.get('id_col', 'id')
    search_type = config.get('search_type', 'flaml')
    search_iters = config.get('search_iters', 100)
    search_time = config.get('search_time', 100)
    estimator_list = config.get('estimator_list', ['lgbm', 'xgboost', 'rf', 'extra_tree'])
    
    train_df = pd.read_csv(train_fn)
    X_train = train_df.drop([target, id_col], axis=1)
    y_train = train_df[target]
    
    results['X_train_head'] = X_train.head().to_dict()
    
    X_test = None
    if test_fn is not None:
        test_df = pd.read_csv(test_fn)
        X_test = test_df.drop([id_col], axis=1)
        results['X_test_head'] = X_test.head().to_dict()
    
    pipe = None
    if search_type == "randomCV":      

        params = {
         'n_jobs': 4,}
        clf = lgb.LGBMClassifier(**params)

        param_grid = {
            'colsample_bytree': uniform(0.6, 0.4),
            'max_depth': randint(5, 30),
            'max_bin': randint(8, 128),
            'num_leaves': randint(16, 512),
            'path_smooth': uniform(0.0, 0.2),
            'n_estimators': randint(25, 1000),
            'min_child_samples': randint(5, 100),
            'learning_rate': uniform(0.001, 0.9),
            'subsample': uniform(0.5, 0.5),
            'reg_alpha': uniform(0.0, 1.0),
            'reg_lambda': uniform(0.0, 1.0),
            'class_weight':['balanced', None],
        }

        pipe = RandomizedSearchCV(clf, param_grid, n_iter=search_iters, n_jobs=10, cv=3, scoring='accuracy', return_train_score=True, verbose=10)
        pipe.fit(X_train, y_train)

        results['cv_results_'] = pipe.cv_results_
        tbl = cv_results_to_df(pipe.cv_results_)
        tbl.to_csv("out/{}-cv_results.csv".format(runname), index=False)
        
    if search_type == "RF":
        clf = RandomForestClassifier(n_jobs=-1, bootstrap=True, n_estimators=1000)

        param_grid = {
            #'n_estimators': randint(150, 500),
            #'n_estimators': randint(4, 1500),
            'max_features': uniform(0.1, 0.9),
            #'min_samples_leaf': randint(1,5),
            #'max_depth': randint(10, 75),
            #'ccp_alpha': uniform(0.0, 0.02),
            'criterion':['gini', 'entropy'],
            'class_weight':['balanced', 'balanced_subsample', None],
        }

        pipe = RandomizedSearchCV(clf, param_grid, n_iter=search_iters, n_jobs=10, cv=2, scoring='accuracy', return_train_score=True, verbose=1)
        pipe.fit(X_train, y_train)
        
        fe = pd.DataFrame({'feature':X.columns, 'importance': pipe.best_estimator_.feature_importances_})
        results['feature_importances_'] = fe.sort_values('importance', ascending=False).to_dict()
        results['oob_score_'] = pipe.best_estimator_.oob_score_
        
        results['max_depths'] = [tree.tree_.max_depth for tree in pipe.best_estimator_.estimators_]  
        results['node_counts'] = [tree.tree_.node_count for tree in pipe.best_estimator_.estimators_]  

        results['cv_results_'] = pipe.cv_results_
        tbl = cv_results_to_df(pipe.cv_results_)
        tbl.to_csv("out/{}-cv_results.csv".format(runname), index=False)
                   

    elif search_type == "flaml":
        
        pipe = AutoML()
        automl_settings = {
            "time_budget": search_time,
            "task": 'classification',
            "log_file_name": "out/flaml-{}.log".format(runname),
            "n_jobs": 20,
            "estimator_list": estimator_list,
            "model_history": True,
            "eval_method": "cv",
            "n_splits": 3,
            "metric": 'accuracy',
            #"metric": custom_metric,
            "log_training_metric": True,
            "verbose": 1,
        }

        results['automl_settings'] = jsonpickle.encode(automl_settings, unpicklable=False, keys=True)

        results['starttime'] = str(datetime.datetime.now())
        
        pipe.fit(X_train, y_train, **automl_settings)
        results['endtime'] = str(datetime.datetime.now())

        results['best_estimator'] = pipe.best_estimator
        results['best_config'] = pipe.best_config
        results['best_loss'] = 1-pipe.best_loss
        results['best_model'] = '{}'.format(str(pipe.model))
        
        print(results['best_loss'])
        
    if pipe is not None and X_test is not None:
        preds = pipe.predict(X_test)
        submission = pd.DataFrame(data={'id': test_df[id_col], 'status_group': preds})
        submission.to_csv("out/{}-stepthom_submission.csv".format(runname), index=False)
        
    results['endtime'] = str(datetime.datetime.now())
    dump_results(runname, results)
    
    print("Run name: {}".format(runname))

if __name__ == "__main__":
    main()
