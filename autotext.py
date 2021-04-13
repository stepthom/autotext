import uuid
import sys
#import logging
import argparse

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

    runname = uuid.uuid4()
    print("Run name: {}".format(runname))

    dumpfile = open('out/{}.txt'.format(runname), "w")

    # Read settings
    with open(args.settings_file) as f:
        config=json.load(f)

    print("Read settings:", file=dumpfile)
    print(json.dumps(config, indent=4, sort_keys=True), file=dumpfile)

    print("Reading data file...", file=dumpfile)
    df = pd.read_csv(config['filename'])
    print("Done. Shape = {}".format(df.shape), file=dumpfile)


    # Kaggle Test Solutions
    df_kaggle = None
    if 'test_filename' in config and config['test_filename'] is not None:
        print("Reading Kaggle test data file...")
        df_kaggle = pd.read_csv(config['test_filename'])
        print("Done. Shape = {}".format(df_kaggle.shape))

    # Define our Target
    target_col = config['target_col']
    text_col = config['text_col']

    # Drop columns that aren't needed
    drop_list=[col for col in df.columns if col not in [target_col, text_col]]
    df = df.drop(drop_list, axis=1)

    print(df.info())
    max_features        = config.get("max_features", None)
    ngram_ranges        = config.get("ngram_ranges", [1, 1])
    sublinear_tfs       = config.get("sublinear_tfs", False)
    stopwords           = config.get("stopwords", None)
    decomposition_types = config.get("decomposition_types", [None])
    n_components        = config.get("n_components", None)

    # regular train/test split
    X_train, X_test, y_train, y_test = train_test_split(df[text_col], df[target_col], test_size=0.2, random_state=1)

    i = 0
    for x in itertools.product(max_features, ngram_ranges, sublinear_tfs, stopwords, decomposition_types, n_components):
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
      features_test = tt.transform(X_test)
      if df_kaggle is not None:
        features_kaggle = tt.transform(df_kaggle[text_col])
      print("Done: shape={}".format(features_train.shape))

      print("Running autosklearn for {} seconds...".format(config['time_left_for_this_task']))


      pipe = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=config['time_left_for_this_task'],
            metric=scorer,
            n_jobs=20,
            seed=42,
            memory_limit=9072,
            include_preprocessors=["no_preprocessing", ],
            exclude_estimators=config['exclude_estimators'],
            )

      pipe = pipe.fit(features_train, y_train)

      print("Done fitting autosklearn.")

      def get_metrics(y_true, y_pred):
          accuracy       = accuracy_score(y_true, y_pred)
          f1             = f1_score(y_true, y_pred, average="macro")
          recall         = recall_score(y_true, y_pred, average="macro")
          precision      = precision_score(y_true, y_pred, average="macro")
          return [accuracy, f1, recall, precision]

      y_pred = pipe.predict(features_test)
      metrics_test = get_metrics(y_test, y_pred)

      metrics_kaggle = []
      if df_kaggle is not None:
        y_pred_kaggle = pipe.predict(features_kaggle)
        metrics_kaggle = get_metrics(df_kaggle[target_col], y_pred_kaggle)
        print('Kaggle results:', file=dumpfile)
        print(classification_report(df_kaggle[target_col], y_pred_kaggle), file=dumpfile)

      lst = [x[0], x[1], x[2], x[3], x[4], x[5]] + metrics_test + metrics_kaggle

      print("Results: {}".format(lst), file=dumpfile)
      res = pd.DataFrame([lst])
      res.to_csv("out/{}-{}.csv".format(runname, i), index=False)

      dumpfile.flush()

      def get_metric_result(cv_results):
          results = pd.DataFrame.from_dict(cv_results)
          results = results[results['status'] == "Success"]
          cols = ['rank_test_scores', 'param_classifier:__choice__', 'mean_test_score']
          cols.extend([key for key in cv_results.keys() if key.startswith('metric_')])
          results.sort_values(by=['rank_test_scores'])
          return results[cols]

      print(classification_report(y_test, y_pred), file=dumpfile)

      print(pipe.sprint_statistics(), file=dumpfile)

      print(pipe.show_models(), file=dumpfile)

      dumpfile.write("#" * 80)
      dumpfile.write("Metric results")
      dumpfile.write(get_metric_result(pipe.cv_results_).to_string(index=False))

      for run_key in pipe.automl_.runhistory_.data:
        print('#########', file=dumpfile)
        print(json.dumps(run_key), file=dumpfile)
        print((pipe.automl_.runhistory_.ids_config[run_key.config_id]), file=dumpfile)
        print(pipe.automl_.runhistory_.data[run_key], file=dumpfile)

      i = i + 1


if __name__ == "__main__":
  main()

