import sys
import argparse
import numpy as np
import json
import pandas as pd
import os
import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from empath import Empath


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "settings_file", help="Path to JSON settings/config file.")
    args = parser.parse_args()

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

    get_lexical = config.get("get_lexical", False)
    get_sentiment = config.get("get_sentiment", False)
    get_empath = config.get("get_empath", False)

    def add_features(df, text_col, get_lexical, get_sentiment, get_empath):

        if get_lexical:
            print("Getting lexical...")
            df['lex_len'] = df[text_col].apply(lambda x: len(x))
            df['lex_syllable_count'] = df[text_col].apply(
                lambda x: textstat.syllable_count(x))
            df['lex_flesch_reading_ease'] = df[text_col].apply(
                lambda x: textstat.flesch_reading_ease(x))
            df['lex_flesch_kincaid_grade'] = df[text_col].apply(
                lambda x: textstat.flesch_kincaid_grade(x))
            df['lex_gunning_fog'] = df[text_col].apply(
                lambda x: textstat.gunning_fog(x))

        if get_sentiment:
            print("Getting sentiment...")
            analyzer = SentimentIntensityAnalyzer()
            df['sent_pos'] = df[text_col].apply(
                lambda x: analyzer.polarity_scores(x)['pos'])
            df['sent_neg'] = df[text_col].apply(
                lambda x: analyzer.polarity_scores(x)['neg'])
            df['sent_neu'] = df[text_col].apply(
                lambda x: analyzer.polarity_scores(x)['neu'])
            df['sent_compound'] = df[text_col].apply(
                lambda x: analyzer.polarity_scores(x)['compound'])

        if get_empath:
            print("Getting empath...")
            empath = Empath()
            _tmp = df[text_col].apply(
                lambda x: empath.analyze(x, normalize=True))
            _tmp = pd.json_normalize(_tmp)
            for col in _tmp.columns:
                df['empath_{}'.format(col)] = _tmp[col]

        return df

    print("Adding features to df... ".format(df.shape))
    df = add_features(df, text_col, get_lexical, get_sentiment, get_empath)
    print("..done: df shape={}".format(df.shape))

    if df_test is not None:
        print("Adding features to test df... ".format(df.shape))
        df_test = add_features(
            df_test, text_col, get_lexical, get_sentiment, get_empath)
        print("..done: df_test shape={}".format(df.shape))

    print("Saving to disk")
    project_name = config.get('project_name', 'unknown_project')
    fname = 'data/{}.csv'.format(project_name)
    fname_test = 'data/{}-test.csv'.format(project_name)
    df.to_csv(fname, index=False)
    if df_test is not None:
        df_test.to_csv(fname_test, index=False)
    print("...done.")


if __name__ == "__main__":
    main()
