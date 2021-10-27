from sklearn.linear_model import LogisticRegressionCV
from flaml import AutoML
from sklearn.preprocessing import StandardScaler
import uuid
import sys
import argparse
import numpy as np

import json

import pandas as pd
import os
import socket

import sklearn
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import datetime

import jsonpickle


import fasttext

def dump_results(runname, results):

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

    with open('out/{}-results.json'.format(runname), 'w') as fp:
        json.dump(results, fp, indent=4)


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
    val_fn = config.get('val_filename', None)
    test_fn = config.get('test_filename', None)
    time = config.get('time_left_for_this_task', 100)
    jobs = config.get('n_jobs', 1)

    print("Running Fasttext  for {} seconds...".format(time))

    model = fasttext.train_supervised(input=train_fn, autotuneValidationFile=val_fn, autotuneDuration=time)

    print("... done running fasttext.")

    print(model.test(test_fn))


    results['endtime'] = str(datetime.datetime.now())

    dump_results(runname, results)
    print("Run name: {}".format(runname))


if __name__ == "__main__":
    main()
