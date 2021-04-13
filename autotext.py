import uuid
import sys
import argparse
import numpy as np

import json

import pandas as pd
import textstat
import itertools

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

import autosklearn.classification
from autosklearn.metrics import balanced_accuracy, precision, recall, f1
import ConfigSpace.read_and_write.json as config_json

from datetime import datetime


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TextTransformer():
    def __init__(self,
                      stop_words='english',
                      strip_accents='unicode',
                      lowercase=True,
                      max_df = 1.0,
                      min_df = 0.0,
                      max_features = 1000,
                      ngram_range = [1,1],
                      sublinear_tf = False,
                      get_lexical = True,
                      get_sentiment = True,
                      decomposition_type = None,
                      n_components = None):


        self.vectorizer = TfidfVectorizer(
                            stop_words=stop_words,
                            strip_accents=strip_accents,
                            lowercase=lowercase,
                            max_df = max_df,
                            min_df = min_df,
                            max_features = max_features,
                            ngram_range = ngram_range,
                            sublinear_tf = sublinear_tf)

        self.decomposition = None
        self.n_components = n_components
        if decomposition_type == 'NMF':
           self.decomposition = NMF(n_components=n_components, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)
        elif decomposition_type == 'SVD':
           self.decomposition = TruncatedSVD(n_components=n_components, random_state=1)

        self.get_lexical = get_lexical
        self.get_sentiment = get_sentiment

        self.analyzer = SentimentIntensityAnalyzer()

    def fit(self, X, y=None):
        self.vectorizer = self.vectorizer.fit(X)
        if self.decomposition  is not None:
            print("in fit, decomp. now={}".format(datetime.now()))
            _dtm = self.vectorizer.transform(X)
            self.decomposition  = self.decomposition.fit(_dtm)
        return self

    def transform(self, X, y=None):
        dtm = self.vectorizer.transform(X)
        f = pd.DataFrame(dtm.toarray(),
                      columns=['bow_{:s}'.format(name) for name in
                                self.vectorizer.get_feature_names()],
                      index=X.index)

        if self.decomposition is not None:
            W = self.decomposition.transform(dtm)
            for i in range(0, self.n_components):
              f['decomp_{:d}'.format(i)] = W[:,i]

        if self.get_lexical:
          f['lex_len'] = X.apply(lambda x: len(x))
          f['lex_syllable_count'] = X.apply(lambda x: textstat.syllable_count(x))
          f['lex_flesch_reading_ease'] = X.apply(lambda x: textstat.flesch_reading_ease(x))
          f['lex_flesch_kincaid_grade'] = X.apply(lambda x: textstat.flesch_kincaid_grade(x))
          f['lex_gunning_fog'] = X.apply(lambda x: textstat.gunning_fog(x))

        if self.get_sentiment:
          f['sent_compound'] = X.apply(lambda x: self.analyzer.polarity_scores(x)['compound'])

        return f


scorer = autosklearn.metrics.make_scorer(
        'f1_score',
        sklearn.metrics.f1_score
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("settings_file", help="Path to JSON settings/config file.")
    args = parser.parse_args()
    runname = str(uuid.uuid4())

    print("Run name: {}".format(runname))

    # This structure will hold all the results and will be dumped to disk.
    results = {}
    results['runname'] = runname

    # This file will hold debugging output
    dumpfile = open('out/{}.log'.format(runname), "w")

    # Read the settings file
    with open(args.settings_file) as f:
        config=json.load(f)

    print("Read settings:", file=dumpfile)
    print(json.dumps(config, indent=4, sort_keys=True), file=dumpfile)

    results['settings'] = config

    print("Reading data file...", file=dumpfile)
    df = pd.read_csv(config['filename'])
    print("Done. Shape = {}".format(df.shape), file=dumpfile)


    # Test file
    df_test = None
    if 'test_filename' in config and config['test_filename'] is not None:
        print("Reading test data file...")
        df_test = pd.read_csv(config['test_filename'])
        print("Done. Shape = {}".format(df_test.shape), file=dumpfile)

    # Define our Target
    target_col = config['target_col']
    text_col = config['text_col']

    # Drop columns that aren't needed
    drop_list=[col for col in df.columns if col not in [target_col, text_col]]
    df = df.drop(drop_list, axis=1)

    max_features        = config.get("max_features", None)
    ngram_ranges        = config.get("ngram_ranges", [1, 1])
    sublinear_tfs       = config.get("sublinear_tfs", False)
    stopwords           = config.get("stopwords", None)
    decomposition_types = config.get("decomposition_types", [None])
    n_components        = config.get("n_components", None)

    # regular train/val split
    X_train, X_val, y_train, y_val = train_test_split(df[text_col], df[target_col], test_size=0.05, random_state=1)

    i = 0
    for x in itertools.product(max_features, ngram_ranges, sublinear_tfs, stopwords, decomposition_types, n_components):

      results[i] = {}
      results[i]['settings'] = x
      print("#" * 80)
      print("Combo: {}".format(x))

      print("Building features...")
      tt = TextTransformer(max_features = x[0],
                           ngram_range = x[1],
                           sublinear_tf = x[2],
                           stop_words = x[3],
                           decomposition_type=x[4],
                           n_components = x[5],
                           get_sentiment=True,
                           get_lexical=True)
      tt = tt.fit(X_train)

      features_train = tt.transform(X_train)
      features_val = tt.transform(X_val)
      if df_test is not None:
        features_test = tt.transform(df_test[text_col])
      print("..done: features_train shape={}".format(features_train.shape))

      time = config.get('time_left_for_this_task', 100)
      jobs = config.get('n_jobs', 1)

      print("Running autosklearn for {} seconds on {} jobs...".format(time, jobs))

      pipe = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time,
            metric=scorer,
            n_jobs=jobs,
            seed=42,
            memory_limit=9072,
            include_preprocessors=["no_preprocessing", ],
            exclude_estimators=config['exclude_estimators'],
            )

      pipe = pipe.fit(features_train, y_train)

      print("... done fitting autosklearn.")

      def get_metrics(y_true, y_pred):
          res = {}
          res['accuracy']       = accuracy_score(y_true, y_pred)
          res['f1']             = f1_score(y_true, y_pred, average="macro")
          res['recall']         = recall_score(y_true, y_pred, average="macro")
          res['precision']      = precision_score(y_true, y_pred, average="macro")
          return res

      y_pred = pipe.predict(features_val)
      results[i]['val_metrics'] = get_metrics(y_val, y_pred)

      if df_test is not None:
        y_pred_test = pipe.predict(features_test)
        df_test['pred'] = y_pred_test
        df_test.to_csv('out/{}-test_pred.csv'.format(runname), index=False)
        if target_col in df_test:
            results[i]['test_metrics'] = get_metrics(df_test[target_col], y_pred_test)
            print('Test results:', file=dumpfile)
            print(classification_report(df_test[target_col], y_pred_test), file=dumpfile)

      print(classification_report(y_val, y_pred), file=dumpfile)
      print(pipe.sprint_statistics(), file=dumpfile)
      print(pipe.show_models(), file=dumpfile)

      #results[i]['cv_results'] = dict(np.ndenumerate(pipe.cv_results_))
      cv = pipe.cv_results_
      cv_df = pd.DataFrame.from_dict(cv)
      cv_df.to_csv('out/{}-cv.csv'.format(runname), index=False)
      results[i]['cv_results'] = {}
      for index, row in cv_df.iterrows():
          results[i]['cv_results'][index] = {}
          results[i]['cv_results'][index]['mean_test_score'] = row['mean_test_score']
          results[i]['cv_results'][index]['mean_fit_time'] = row['mean_fit_time']
          results[i]['cv_results'][index]['rank_test_scores'] = row['rank_test_scores']
          results[i]['cv_results'][index]['status'] = row['status']
          results[i]['cv_results'][index]['params'] = row['params']
      #for key, value in cv.items():
          #print("key={}, value={}".format(key, value))
          #print(type(key))
          #print(type(value))
          #results[i]['cv_results'][key] = value

      dumpfile.flush()
      with open('out/{}-results.json'.format(runname), 'w') as fp:
        json.dump(results, fp, indent=4)

      i = i + 1


if __name__ == "__main__":
  main()

