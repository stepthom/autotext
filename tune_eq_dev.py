import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import comet_ml at the top of your file
from comet_ml import Experiment
import uuid
import argparse
import numpy as np
import pandas as pd
import os
import sys

import json
import socket
import datetime

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

from flaml import AutoML

from functools import partial

from SteveHelpers import dump_json, get_data_types
from SteveHelpers import SteveCorexWrapper
from SteveHelpers import SteveNumericNormalizer
from SteveHelpers import SteveAutoFeatLight
from SteveHelpers import SteveFeatureDropper
from SteveHelpers import SteveFeatureTyper
from SteveHelpers import SteveEncoder
from SteveHelpers import SteveNumericCapper
from SteveHelpers import SteveMissingIndicator
from SteveHelpers import SteveNumericImputer
from SteveHelpers import SteveCategoryImputer
from SteveHelpers import SteveCategoryCoalescer

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--enable-comet', type=int, default=1)
    parser.add_argument('--geo-id-set', type=int, default=3)
    parser.add_argument('--algo-set', type=int, default=1)
    parser.add_argument('--n-hidden', type=int, default=4)
    parser.add_argument('--dim-hidden', type=int, default=2)
    parser.add_argument('--smooth-marginals', type=int, default=0)
    parser.add_argument('--min-sample-leaf', type=int, default=5)
    parser.add_argument('--time-budget', type=int, default=60)
    parser.add_argument('--smoothing', type=float, default=10.0)
    parser.add_argument('--autofeat', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--sample-frac', type=float, default=1.0)
    parser.add_argument('--ensemble', type=int, default=0)
    parser.add_argument('--run-type', type=str, default="flaml")
    parser.add_argument('--metric', type=str, default="micro_f1")
    parser.add_argument('--cat-encoder', type=int, default=1)

    args = parser.parse_args()

    print("Command line arguments:")
    print(vars(args))

    id_col = 'building_id'
    target_col = 'damage_grade'
    out_dir = 'earthquake/out'
    data_id = '000'

    # Create an experiment with your api key
    exp=None
    if args.enable_comet == 1:
        exp = Experiment(
            project_name="eq1",
            workspace="stepthom",
            parse_args=False,
            auto_param_logging=False,
            auto_metric_logging=False,
            log_graph=False
        )

    train_fn =  'earthquake/earthquake_train.csv'
    test_fn =  'earthquake/earthquake_test.csv'
    train_df  = pd.read_csv(train_fn)
    test_df  = pd.read_csv(test_fn)
    if args.sample_frac < 1.0:
        train_df  = train_df.sample(frac=args.sample_frac, random_state=3).reset_index(drop=True)


    geo_id_set = []
    if args.geo_id_set == 1:
        geo_id_set = ['geo_level_1_id']
    elif args.geo_id_set == 2:
        geo_id_set = ['geo_level_1_id', 'geo_level_2_id']
    else:
        geo_id_set = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']

    estimator_list = ['lgbm']
    if args.algo_set == 1:
        estimator_list = ['lgbm']
    elif args.algo_set == 2:
        estimator_list = ['xgboost']
    elif args.algo_set == 3:
        estimator_list = ['catboost']
    elif args.algo_set == 4:
        estimator_list = ['rf']
    elif args.algo_set == 5:
        estimator_list = ['extra_tree']

    X = train_df.drop([id_col, target_col], axis=1)
    y = train_df[target_col]
    X_test = test_df.drop([id_col], axis=1)

    prep1 = Pipeline([
        ('typer', SteveFeatureTyper(cols=geo_id_set, typestr='category'))
    ])

    prep1.fit(X)
    X = prep1.transform(X)
    X_test = prep1.transform(X_test)

    cat_cols, num_cols, bin_cols, float_cols, date_cols = get_data_types(X, id_col, target_col)

    steps = []

    if False and args.n_hidden >= 1:
        steps.append(('corex', SteveCorexWrapper(bin_cols, n_hidden=args.n_hidden)))
        steps.append(('dropper', SteveFeatureDropper(bin_cols)))

    if args.cat_encoder == 1:
        pass
    elif args.cat_encoder == 2:
        _cat_cols = list(set(cat_cols) - set(geo_id_set))
        steps.append(('cat_encoder', SteveEncoder(cols=_cat_cols,
                      encoder=OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, dtype=np.int32))))
        steps.append(('dropper', SteveFeatureDropper(_cat_cols)))
    elif args.cat_encoder == 3:
        _cat_cols = list(set(cat_cols) - set(geo_id_set))
        steps.append(('cat_encoder', SteveEncoder(cols=_cat_cols,
                       encoder=OneHotEncoder(handle_unknown='ignore', sparse=False, dtype=np.int32))))
        steps.append(('dropper', SteveFeatureDropper(_cat_cols)))

    steps.append(('num_capper', SteveNumericCapper(num_cols=['age'], max_val=30)))

    if args.autofeat == 1:
        steps.append(('num_autofeat', SteveAutoFeatLight(float_cols)))

    if args.normalize == 1:
        steps.append(('num_normalizer', SteveNumericNormalizer(float_cols, drop_orig=True)))

    print(steps)

    preprocessor = Pipeline(steps)

    preprocessor.fit(X)
    X = preprocessor.transform(X)
    X_test = preprocessor.transform(X_test)

    print("X head:")
    print(X.head().T)
    print("X dtypes:")
    print(X.dtypes)
    print("X_test head:")
    print(X_test.head().T)

    runname = ""
    if exp:
        runname = exp.get_key()
    else:
        runname = str(uuid.uuid4())
    run_fn = os.path.join(out_dir, "tune_eq_{}.json".format(runname))
    print("tune_eq: Run name: {}".format(runname))
    
    def f1_micro_soft(_y_true, y_pred_proba):
    
        # Convert y_true to OHE
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y_true = enc.fit_transform(np.array(_y_true).reshape(-1, 1))

        #print(y_true)
        #print(np.sum(y_true, axis=0))

        tp = np.sum(y_pred_proba * y_true, axis=0)
        fp = np.sum(y_pred_proba * (1 - y_true), axis=0)
        fn = np.sum((1 - y_pred_proba) * y_true, axis=0)
        soft_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        micro_soft_f1 = np.average(soft_f1, weights=np.sum(y_true, axis=0)) 
        macro_soft_f1 = np.average(soft_f1, weights=None) # average on all labels

        #print(tp)
        #print(fp)
        #print(fn)
        #print(soft_f1)
        #print(micro_soft_f1)
        #print(macro_soft_f1)
        return macro_soft_f1
        
    
    def custom_metric(X_val, y_val, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, return_metric="micro_f1"):
        from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, classification_report
        import time
        
        start = time.time()
        y_pred_proba = estimator.predict_proba(X_val)
        y_pred = np.argmax(y_pred_proba,axis=1)
        y_pred = estimator.predict(X_val)
        pred_time = (time.time() - start) / len(X_val)
        
        class_report = classification_report(y_val, y_pred, sample_weight=weight_val, output_dict=True)
        class_report_flat = pd.json_normalize(class_report, sep='_').to_dict(orient='records')[0]
        class_report_flat = {'val_metric_' + str(key): val for key, val in class_report_flat.items()}
        
        custom_metrics = {
            'val_metric_log_loss':  log_loss(y_val, y_pred_proba, labels=labels, sample_weight=weight_val),
            'val_metric_micro_f1':  f1_score(y_val, y_pred, sample_weight=weight_val, average="micro"),
            'val_metric_macro_f1':  f1_score(y_val, y_pred, sample_weight=weight_val, average="macro"),
            'val_metric_weighted_f1':  f1_score(y_val, y_pred, sample_weight=weight_val, average="weighted"),
            'val_metric_roc_auc':  roc_auc_score(y_val, y_pred_proba, sample_weight=weight_val, multi_class="ovo"),
            'val_metric_micro_f1_soft':  f1_micro_soft(y_val, y_pred_proba),
        }
        custom_metrics.update(class_report_flat)
        
        y_pred_proba = estimator.predict_proba(X_train)
        y_pred = np.argmax(y_pred_proba,axis=1)
        y_pred = estimator.predict(X_train)
        
        class_report = classification_report(y_train, y_pred, sample_weight=weight_train, output_dict=True)
        class_report_flat = pd.json_normalize(class_report, sep='_').to_dict(orient='records')[0]
        class_report_flat = {'train_metric_' + str(key): val for key, val in class_report_flat.items()}
       
        custom_metrics.update({
            'train_metric_log_loss':  log_loss(y_train, y_pred_proba, labels=labels, sample_weight=weight_train),
            'train_metric_micro_f1':  f1_score(y_train, y_pred, sample_weight=weight_train, average="micro"),
            'train_metric_macro_f1':  f1_score(y_train, y_pred, sample_weight=weight_train, average="macro"),
            'train_metric_weighted_f1':  f1_score(y_train, y_pred, sample_weight=weight_train, average="weighted"),
            'train_metric_roc_auc':  roc_auc_score(y_train, y_pred_proba, sample_weight=weight_train, multi_class="ovo"),
            'train_metric_micro_f1_soft':  f1_micro_soft(y_train, y_pred_proba),
        })
        custom_metrics.update(class_report_flat)
        
        custom_metrics.update({
            "pred_time": pred_time,
        })
      
        val_loss = None
        return_metric_name = "val_metric_{}".format(return_metric)
        if return_metric_name == "val_metric_log_loss":
            val_loss = custom_metrics[return_metric_name]
        else:
            val_loss = 1 - custom_metrics[return_metric_name]
        return val_loss, custom_metrics
   
    # The below is so we can re-use the custom_metric code but return different things based on the command line arg "metric"
    custom_metric_flaml = partial(custom_metric, return_metric=args.metric)


    if exp is not None:
        exp.log_other('train_fn', train_fn)
        exp.log_other('test_fn', test_fn)
        exp.log_table('X_head.csv', X.head())
        exp.log_table('y_head.csv', y.head())
        exp.log_parameters(vars(args))
        exp.log_asset('SteveHelpers.py')

    starttime = datetime.datetime.now()
    feat_imp = None
    if args.run_type == "flaml":
        os.environ['OS_STEVE_MIN_SAMPLE_LEAF'] = str(args.min_sample_leaf)
        os.environ['OS_STEVE_SMOOTHING'] = str(args.smoothing)
       
        automl_settings = {
            "time_budget": args.time_budget,
            "task": 'classification',
            "n_jobs": 5,
            "estimator_list": estimator_list,
            "eval_method": "cv",
            "n_splits": 3,
            "metric": custom_metric_flaml,
            "ensemble": False,
        }
        automl_config = {
            "comet_exp": exp,
            "verbose": 1,
            "log_training_metric": True,
            "model_history": False,
            "log_file_name": "logs/flaml-{}.log".format(runname),
        }

        print("automl_settings:")
        print(automl_settings)
        print("automl_config:")
        print(automl_config)

        # Log some things before fit(), to more-easily monitor runs
        if exp is not None:
            exp.log_parameters(automl_settings)
            exp.log_parameter("metric_name", args.metric)

        clf = AutoML()
        clf.fit(X, y, **automl_config, **automl_settings)
        best_loss = clf.best_loss

        print("Best model")
        bm = clf.best_model_for_estimator(clf.best_estimator)
        print(bm.model)
        feature_names = bm.feature_names_
        feat_imp =  pd.DataFrame({'Feature': feature_names, 'Importance': bm.model.feature_importances_}).sort_values('Importance', ascending=False)
        print("Feature importances")
        print(feat_imp.head())

    elif args.run_type == "opt":
        from lightgbm import LGBMClassifier
        from sklearn.model_selection import train_test_split
        import category_encoders as ce
        from category_encoders.wrapper import PolynomialWrapper
        from sklearn.metrics import classification_report, f1_score
        
        params = {
            "n_estimators":1239,
             "num_leaves":772,
             "min_child_samples":42,
             "learning_rate":0.008946391199324397,
             "subsample":0.8627473025492609,
             "colsample_bytree":0.28502068330476027,
             "reg_alpha":0.025620353309563408,
             "reg_lambda":8.122050782004282
        }
        
        
        def calc_metrics(probas, weights):
            #print(probas)
        
            probas_n = np.multiply(probas, weights)
            preds = np.argmax(probas_n, axis=1) + 1
            #print(classification_report(y_val, preds))
            return f1_score(y_val, preds, average="micro")
        
        res = []
        for cv_iter in range(10):
            print("CV {}".format(cv_iter))
        
            X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=cv_iter, stratify=y)
            enc =  ce.wrapper.PolynomialWrapper(
                    ce.target_encoder.TargetEncoder(handle_unknown="value", handle_missing="value", min_samples_leaf=3, smoothing=0.1))

            clf = LGBMClassifier(**params)
            pipe = Pipeline([
                ("cat_enc", enc),
                ("clf", clf),
            ])

            pipe.fit(X_train, y_train)

            probas = pipe.predict_proba(X_val)
            baseline_f1 = calc_metrics(probas, [1, 1, 1])
            res.append({ 'i': 1, 'j': 1, 'k': 1, 'cv_iter': cv_iter, 'f1': baseline_f1, 'diff': 0 })

            for i in [1]:
                for j in np.linspace(0.5, 1.5, num=15):
                    for k in np.linspace(0.5, 1.5, num=15):
                        f1 =  calc_metrics(probas, [i, j, k])
                        res.append({
                            'i': i,
                            'j': j,
                            'k': k,
                            'cv_iter': cv_iter,
                            'f1': f1,
                            'diff': f1-baseline_f1,
                        })
        df = pd.DataFrame(res)
        df = df.sort_values('diff', ascending=False)
        print(df.head(15))
        print(df.tail(5))
        df.to_csv('logs/heatmap-all-{}.csv'.format(cv_iter), index=False)
        grp = df.groupby(['i','j','k']).agg({'f1': 'describe', 'diff': 'describe'})
        print(grp)


        return
    
    else:
        pass

    endtime = datetime.datetime.now()
    duration = (endtime - starttime).seconds

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

    results = {}
    results['runname'] = runname
    results['preds_fn'] = preds_fn
    results['probas_fn'] = probas_fn
    results['endtime'] = str(endtime)
    results['algo_set'] = args.algo_set
    results['best_loss'] =  best_loss
    dump_json(run_fn, results)

    if exp is not None:
        exp.log_metric("duration", duration)

        if hasattr(clf, "best_config_train_train"):
            exp.log_metric("best_config_train_time", clf.best_config_train_time)

        if feat_imp is not None:
            exp.log_table('feat_imp.csv', feat_imp)

        if hasattr(clf, "best_config"):
            exp.log_text(clf.best_config)

        if hasattr(clf, "best_estimator"):
            exp.log_text(clf.best_model_for_estimator(clf.best_estimator).model)

if __name__ == "__main__":
    main()
