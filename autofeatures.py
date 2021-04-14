import uuid
import sys
import argparse
import numpy as np

import json

import pandas as pd
import textstat
import itertools
import os

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TextTransformer():
    def __init__(self,
                 stop_words='english',
                 strip_accents='unicode',
                 lowercase=True,
                 max_df=1.0,
                 min_df=0.0,
                 max_features=1000,
                 ngram_range=[1, 1],
                 sublinear_tf=False,
                 get_lexical=True,
                 get_sentiment=True,
                 decomposition_type=None,
                 n_components=None):

        self.vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            strip_accents=strip_accents,
            lowercase=lowercase,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf)

        self.decomposition = None
        self.n_components = n_components
        if decomposition_type == 'NMF':
            self.decomposition = NMF(n_components=n_components, random_state=1,
                                     init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)
        elif decomposition_type == 'SVD':
            self.decomposition = TruncatedSVD(
                n_components=n_components, random_state=1)

        self.get_lexical = get_lexical
        self.get_sentiment = get_sentiment

        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        self.vectorizer = self.vectorizer.fit(X)
        if self.decomposition is not None:
            _dtm = self.vectorizer.transform(X)
            self.decomposition = self.decomposition.fit(_dtm)
        return self

    def transform(self, X, y=None):
        dtm = self.vectorizer.transform(X)
        f = pd.DataFrame(dtm.toarray(),
                         columns=['bow_{:s}'.format(name.replace(" ", "_")) for name in
                                  self.vectorizer.get_feature_names()],
                         index=X.index)

        if self.decomposition is not None:
            W = self.decomposition.transform(dtm)
            for i in range(0, self.n_components):
                f['decomp_{:d}'.format(i)] = W[:, i]

        if self.get_lexical:
            f['lex_len'] = X.apply(lambda x: len(x))
            f['lex_syllable_count'] = X.apply(
                lambda x: textstat.syllable_count(x))
            f['lex_flesch_reading_ease'] = X.apply(
                lambda x: textstat.flesch_reading_ease(x))
            f['lex_flesch_kincaid_grade'] = X.apply(
                lambda x: textstat.flesch_kincaid_grade(x))
            f['lex_gunning_fog'] = X.apply(lambda x: textstat.gunning_fog(x))

        if self.get_sentiment:
            f['sent_compound'] = X.apply(
                lambda x: self.analyzer.polarity_scores(x)['compound'])

        if y is not None:
            f['label'] = y

        return f



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

    # Test file
    df_test = None
    if 'test_filename' in config and config['test_filename'] is not None:
        print("Reading test data file...")
        df_test = pd.read_csv(config['test_filename'])
        print("Done. Shape = {}".format(df_test.shape))

    # Define our Target
    target_col = config['target_col']
    text_col = config['text_col']

    # Drop columns that aren't needed
    drop_list = [col for col in df.columns if col not in [
        target_col, text_col]]
    df = df.drop(drop_list, axis=1)

    max_features = config.get("max_features", None)
    ngram_ranges = config.get("ngram_ranges", [1, 1])
    sublinear_tfs = config.get("sublinear_tfs", False)
    stopwords = config.get("stopwords", None)
    decomposition_types = config.get("decomposition_types", [None])
    n_components = config.get("n_components", None)

    # regular train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        df[text_col], df[target_col], test_size=0.05, random_state=1)

    i = 0
    for x in itertools.product(max_features, ngram_ranges, sublinear_tfs, stopwords, decomposition_types, n_components):

        print("#" * 80)
        print("Combo: {}".format(x))

        print("Building features...")
        tt = TextTransformer(max_features=x[0],
                             ngram_range=x[1],
                             sublinear_tf=x[2],
                             stop_words=x[3],
                             decomposition_type=x[4],
                             n_components=x[5],
                             get_sentiment=True,
                             get_lexical=True)
        tt = tt.fit(X_train)

        features_train = tt.transform(X_train, y_train)
        features_val = tt.transform(X_val, y_val)
        if df_test is not None:
            features_test = tt.transform(df_test[text_col], df_test[target_col])
        print("..done: features_train shape={}".format(features_train.shape))

        print("Saving features to disk")
        project_name = config.get('project_name', 'unknown_project')
        data_combo_name = "{}-{}-{}-{}-{}-{}".format(x[0], x[1], x[2], x[3], x[4], x[5], True, True)
        print('Data combo name: {}'.format(data_combo_name))
        fname_train = 'data/{}-{}-features_train.csv'.format(project_name, data_combo_name)
        fname_val   = 'data/{}-{}-features_val.csv'.format(project_name, data_combo_name)
        fname_test  = 'data/{}-{}-features_test.csv'.format(project_name, data_combo_name)
        features_train.to_csv(fname_train, index=False)
        features_val.to_csv(fname_val, index=False)
        if features_test is not None:
            features_test.to_csv(fname_test, index=False)
        print("...done.")


        i = i + 1


if __name__ == "__main__":
    main()
