import uuid
import sys
import argparse

import json

import pandas as pd
import os
from sklearn.model_selection import train_test_split

from datetime import datetime


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "settings_file", help="Path to JSON settings/config file.")
    args = parser.parse_args()
    runname = str(uuid.uuid4())

    print("Run name: {}".format(runname))

    # Read the settings file
    with open(args.settings_file) as f:
        config = json.load(f)

    print("Reading data file...")
    df = pd.read_csv(config['filename'])
    print("Done. Shape = {}".format(df.shape))

    project_name = config.get('project_name', 'unknown_project')

    # Test file
    test = None
    if 'test_filename' in config and config['test_filename'] is not None:
        print("Reading test data file...")
        test = pd.read_csv(config['test_filename'])
        print("Done. Shape = {}".format(test.shape))

    # Define our Target
    target_col = config['target_col']
    text_col = config['text_col']

    # regular train/val split
    train, val = train_test_split(df, test_size=0.20, random_state=1)

    print("Writing fasttext format...")
    with open('data/{}-train.txt'.format(project_name),'w') as f:
        for index, row in train.iterrows():
            f.write('__label__{} {}\n'.format(row[target_col], row[text_col]))
    with open('data/{}-val.txt'.format(project_name),'w') as f:
        for index, row in val.iterrows():
            f.write('__label__{} {}\n'.format(row[target_col], row[text_col]))
    with open('data/{}-test.txt'.format(project_name),'w') as f:
        for index, row in test.iterrows():
            f.write('__label__{} {}\n'.format(row[target_col], row[text_col]))


if __name__ == "__main__":
    main()
