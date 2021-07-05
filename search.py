import uuid
import sys
import argparse
import numpy as np
import pandas as pd

import json
import socket
import datetime

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
import lightgbm as lgb
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, StackingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

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
    
    parser.add_argument('--use-ensemble', dest='use_ensemble', default=False, action='store_true')
    parser.add_argument('--use-sample', dest='use_sample', default=False, action='store_true')
    parser.add_argument('-s', '--search-type', nargs='?', default=None)
    
    args = parser.parse_args()
        
    runname = str(uuid.uuid4())

    print("Run name: {}".format(runname))

    # This structure will hold all the results and will be dumped to disk.
    results = {}
    results['runname'] = runname
    results['starttime'] = str(datetime.datetime.now())
    results['hostname'] = socket.gethostname()
    
    # Read the settings file
    print("DEBUG: Reading settings file")
    with open(args.settings_file) as f:
        config = json.load(f)
    
    train_fn = config.get('train_filename', None)
    test_fn = config.get('test_filename', None)
    target = config.get('target', 'status_group')
    id_col = config.get('id_col', 'id')
    
    search_type = args.search_type
    if search_type is None:
        search_type = config.get('search_type', 'flaml')
    search_iters = config.get('search_iters', 100)
    search_time = config.get('search_time', 100)
    estimator_list = config.get('estimator_list', ['lgbm', 'xgboost', 'rf', 'extra_tree'])
    drop_cols = config.get('drop_cols', None)
    
    results['use_ensemble'] = args.use_ensemble
    results['use_sample'] = args.use_sample
    results['search_type'] = search_type
    results['settings'] = config
    
    if False:
        drop_cols = ['longitude_missing', 'population_missing', 'mwanza_dist', 'gps_height_missing', 'permit_missing', 
                  'timediff', 'extraction_type_group_non functional', 'quarter_date_recorded', 'daressallam_dist', 
                  'extraction_type_class_non functional', 
                  'year_date_recorded',
                  'source_class_functional needs repair', 'dayofyear_date_recorded', 'installer_missing', 'quality_group_non functional', 
                  'extraction_type_group_functional needs repair']
    
    print("DEBUG: Reading training data")
    train_df = pd.read_csv(train_fn)
    if drop_cols is not None:
        train_df = train_df.drop(drop_cols, axis=1)
        
    if args.use_sample:
        print("DEBUG: Taking sample")
        train_df = train_df.sample(frac=0.2, random_state=42)
        
    X_train = train_df.drop([target, id_col], axis=1)
    y_train = train_df[target]
    
    results['X_train_head'] = X_train.head().to_dict()
    
    X_test = None
    if test_fn is not None:
        print("DEBUG: Reading testing data")
        test_df = pd.read_csv(test_fn)
        if drop_cols is not None:
            test_df = test_df.drop(drop_cols, axis=1)
        X_test = test_df.drop([id_col], axis=1)
        results['X_test_head'] = X_test.head().to_dict()
    
    pipe = None
    
    if search_type == "mljar":
        from supervised.automl import AutoML # mljar-supervised
        pipe = AutoML(mode="Compete", eval_metric="accuracy")
        pipe.fit(X_train, y_train)
    
    elif search_type == "hist":
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        from sklearn.ensemble import HistGradientBoostingClassifier

        cat_features=None
        
        if True:
            cat_feature_names= ['enc_0','enc_1','enc_2','enc_3',
                       'enc_4','enc_5','enc_6','enc_7','enc_8','enc_9','enc_10',
                       'enc_11','enc_12','enc_13','enc_14','enc_15','enc_16',
                       'enc_17','enc_18','enc_19','enc_20','enc_21','enc_22']
            
            cat_features = []
            for cat_feature_name in cat_feature_names:
                cat_features.append(list(X_train.columns).index(cat_feature_name))
            print("DEBUG: cat_features: {}".format(cat_features))

        clf = HistGradientBoostingClassifier(categorical_features = cat_features, random_state=42)
        
        param_grid = {
            'max_iter': randint(500, 1500),
            'max_leaf_nodes': randint(31, 255),
            'max_bins': randint(102, 255),
            'learning_rate': uniform(0.001, 0.05),

        }

        pipe = RandomizedSearchCV(clf, param_grid, n_iter=search_iters, n_jobs=10, cv=2, scoring='accuracy', return_train_score=True, verbose=1)
        pipe.fit(X_train, y_train)

        results['cv_results_'] = pipe.cv_results_
        tbl = cv_results_to_df(pipe.cv_results_)
        tbl.to_csv("out/{}-cv_results.csv".format(runname), index=False)
        
    elif search_type == "RF":
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
        
    elif search_type == "gpc":
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.model_selection import ShuffleSplit
        
        pipe = GaussianProcessClassifier(kernel=None, n_jobs=10, warm_start=True, copy_X_train=False, n_restarts_optimizer=10, random_state=42)
        
        print("DEBUG: cross_val_score")
        cv = ShuffleSplit(5, test_size=0.2, train_size=0.2, random_state=0)
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, n_jobs=1, verbose=2)
        results['scores'] = scores
        results['mean_scores'] = np.mean(scores)
        print("DEBUG: scores: {}".format(scores))
        
        print("DEBUG: fit on full")
        pipe.fit(X_train, y_train)
        
        
    elif search_type == "stack":
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        from sklearn.ensemble import HistGradientBoostingClassifier
        
        estimators = [
            
            #Best tuned for 75 mestimate
            #('lgbm1', lgb.LGBMClassifier(colsample_bytree=0.28675389617274555, learning_rate=0.010452067895102068, 
            #                             max_bin=1023, min_child_samples=30, n_estimators=366, n_jobs=3, 
            #                             num_leaves=24644, objective='multiclass', reg_alpha=0.0009765625, 
            #                             reg_lambda=0.5525418150514663, subsample=0.676715616413845)),
            #('rf1', RandomForestClassifier(n_estimators=2048, max_features=0.210054, criterion="gini", random_state=42, n_jobs=3)),
            #('hist', HistGradientBoostingClassifier(learning_rate=0.0512380, max_bins=137, max_iter=947, max_leaf_nodes=120, random_state=42)),
            
            #Best tuned for 100 mestimate
            ('lgbm1', lgb.LGBMClassifier(colsample_bytree=0.28675389617274555, learning_rate=0.010452067895102068, 
                                         max_bin=1023, min_child_samples=30, n_estimators=366, n_jobs=3, 
                                         num_leaves=24644, objective='multiclass', reg_alpha=0.0009765625, 
                                         reg_lambda=0.5525418150514663, subsample=0.676715616413845)),
            ('rf1', RandomForestClassifier(n_estimators=2048, max_features=0.210054, criterion="gini", random_state=42, n_jobs=3)),
            ('hist', HistGradientBoostingClassifier(learning_rate=0.0512380, max_bins=137, max_iter=947, max_leaf_nodes=120, random_state=42)),
        ]
        
        #0.051238087074071514,134,947,120
        
        fe = LogisticRegression()
        #fe = RandomForestClassifier()
       
        pipe = StackingClassifier(estimators=estimators, final_estimator=fe, cv=5, n_jobs=2)
        
        #print("DEBUG: cross_val_score")
        #scores = cross_val_score(pipe, X_train, y_train, cv=2, n_jobs=1, verbose=2)
        #results['scores'] = scores
        #results['mean_scores'] = np.mean(scores)
        
        print("DEBUG: fit on full")
        pipe.fit(X_train, y_train)

    elif search_type == "flaml":
        
        from flaml import AutoML
        
        pipe = AutoML()
        automl_settings = {
            "time_budget": search_time,
            "task": 'classification',
            "log_file_name": "out/flaml-{}.log".format(runname),
            "n_jobs": 20,
            "estimator_list": estimator_list,
            "model_history": True,
            #"eval_method": "cv",
            #"n_splits": 3,
            "metric": 'accuracy',
            #"metric": custom_metric,
            "log_training_metric": True,
            "verbose": 1,
            "ensemble": args.use_ensemble,
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
        
    elif search_type == "autosklearn":
        
        import autosklearn.classification
        
        automl_settings = {
            "time_left_for_this_task": search_time,
            "n_jobs": 20,
            #"include_estimators": estimator_list,   
            "memory_limit": 20 * 1024,
        }
        
        pipe = autosklearn.classification.AutoSklearnClassifier(**automl_settings)
        pipe.fit(X_train, y_train)
        
        results['cv_results_'] = pipe.cv_results_
        #tbl = cv_results_to_df(pipe.cv_results_)
        #tbl.to_csv("out/{}-cv_results.csv".format(runname), index=False)
        
    elif search_type == "svm":
        from sklearn.svm import SVC, LinearSVC
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import reciprocal, uniform
        from sklearn.model_selection import ShuffleSplit
        
        clf = make_pipeline(StandardScaler(), LinearSVC())

        param_grid = {
            #"svc__gamma": reciprocal(0.001, 0.1),
            "linearsvc__C": uniform(1, 5),
            #"svc__kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
            #"svc__kernel": ['linear'],
            "linearsvc__dual": [True, False],
            "linearsvc__penalty": ['l1', 'l2'],
            "linearsvc__class_weight": ['balanced', None],
        }

        cv = ShuffleSplit(2, test_size=0.2, train_size=0.7, random_state=0)
        pipe = RandomizedSearchCV(clf, param_grid, n_iter=search_iters, n_jobs=3, cv=cv, scoring='accuracy', return_train_score=True, verbose=1)
        pipe.fit(X_train, y_train)
        
        results['cv_results_'] = pipe.cv_results_
        tbl = cv_results_to_df(pipe.cv_results_)
        tbl.to_csv("out/{}-cv_results.csv".format(runname), index=False)
        
    elif search_type == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline, Pipeline
        from sklearn.preprocessing import StandardScaler
        from scipy.stats import reciprocal, uniform
        from sklearn.model_selection import ShuffleSplit
        
        clf = Pipeline(steps=[('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])

        param_grid = {
            "clf__n_neighbors": randint(3, 30),
            "clf__weights": ['uniform', 'distance'],
            "clf__p": [1, 2, 3],
        }

        cv = ShuffleSplit(2, test_size=0.10, train_size=0.20, random_state=0)
        pipe = RandomizedSearchCV(clf, param_grid, n_iter=search_iters, n_jobs=5, cv=cv, scoring='accuracy', return_train_score=True, verbose=1)
        pipe.fit(X_train, y_train)
        
        results['cv_results_'] = pipe.cv_results_
        tbl = cv_results_to_df(pipe.cv_results_)
        tbl.to_csv("out/{}-cv_results.csv".format(runname), index=False)
        
    if pipe is not None and X_test is not None:
        preds = pipe.predict(X_test)
        submission = pd.DataFrame(data={'id': test_df[id_col], 'status_group': preds})
        submission.to_csv("out/{}-stepthom_submission.csv".format(runname), index=False)
        
        probas = pipe.predict_proba(X_test)
        probas_df = pd.DataFrame(probas, columns=pipe.classes_)
        probas_df[id_col] = test_df[id_col]
        probas_df = probas_df[ [id_col] + [ col for col in probas_df.columns if col != id_col ] ]
        probas_df.to_csv("out/{}-probas.csv".format(runname), index=False)
        
    results['endtime'] = str(datetime.datetime.now())
    dump_results(runname, results)
    
    print("Run name: {}".format(runname))

if __name__ == "__main__":
    main()
