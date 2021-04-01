import uuid
import sys
import logging
fmt = '%(asctime)s %(levelname)-8s %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=fmt,
                    datefmt='%m-%d %H:%M',
                    handlers=[
                        logging.FileHandler("{}.log".format(uuid.uuid4())),
                        logging.StreamHandler(sys.stdout)
                        ])


import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import textstat
import itertools

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GroupShuffleSplit, TimeSeriesSplit
import sklearn
import autosklearn.classification
from autosklearn.metrics import balanced_accuracy, precision, recall, f1




def print_metrics(pipe, X_test, y_test):
  y_pred = pipe.predict(X_test)

  print(classification_report(y_test, y_pred))

  df_eval=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
  print(pd.crosstab(df_eval['Actual'],df_eval['Predicted']))



def get_text_features(df,
                      text_feature,
                      stop_words='english',
                      strip_accents='unicode',
                      lowercase=True,
                      max_df = 1.0,
                      min_df = 0.0,
                      max_features = 1000,
                      ngram_range = [1,1],
                      sublinear_tf = False,
                      get_lexical = True,
                      n_components = None
                      ):

  vectorizer = TfidfVectorizer(
                            stop_words=stop_words,
                            strip_accents=strip_accents,
                            lowercase=lowercase,
                            max_df = max_df,
                            min_df = min_df,
                            max_features = max_features,
                            ngram_range = ngram_range,
                            sublinear_tf = sublinear_tf)


  vectorizer = vectorizer.fit(df[text_feature])
  dtm = vectorizer.transform(df[text_feature])
  f = pd.DataFrame(dtm.toarray(),
                      columns=['bow_{:s}'.format(name) for name in
                                vectorizer.get_feature_names()],
                      index=df.index)

  if n_components is not None:

    nmf = NMF(n_components=n_components, random_state=1, init='nndsvda', solver='mu', alpha=.1, l1_ratio=.5)
    W = nmf.fit_transform(dtm)
    for i in range(0, n_components):
      f['nmf_{:d}'.format(i)] = W[:,i]

  if get_lexical:
    f['lex_len'] = df[text_feature].apply(lambda x: len(x))
    f['lex_syllable_count'] = df[text_feature].apply(lambda x: textstat.syllable_count(x))
    f['lex_flesch_reading_ease'] = df[text_feature].apply(lambda x: textstat.flesch_reading_ease(x))
    f['lex_flesch_kincaid_grade'] = df[text_feature].apply(lambda x: textstat.flesch_kincaid_grade(x))
    f['lex_gunning_fog'] = df[text_feature].apply(lambda x: textstat.gunning_fog(x))
    #f['lex_coleman_liau_index'] = df[text_feature].apply(lambda x: textstat.coleman_liau_index(x))

  return f





scorer = autosklearn.metrics.make_scorer(
        'f1_score',
        sklearn.metrics.f1_score
    )

def get_auto(X, y):

  feature_names = X.columns

  # regular train/test split
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


  pipeline_auto = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=60, #2*60*60,
        metric=scorer,
        n_jobs=10,
        seed=42,
        memory_limit=5072,
        exclude_estimators=['k_nearest_neighbors',  'mlp',
                    'gaussian_nb', 'bernoulli_nb', 'multinomial_nb',
                    'decision_tree', 'lda',
                    'liblinear_svc', 'libsvm_svc', 'qda'],
        )

  pipeline_auto = pipeline_auto.fit(X_train, y_train, X_test, y_test)

  logging.info("#" * 80)
  print('metrics')
  print_metrics(pipeline_auto, X_test, y_test)

  #print("#" * 80)
  #print('statistics')
  #print(pipeline_auto.sprint_statistics())

  #print("#" * 80)
  #print("show_models")
  #print(pipeline_auto.show_models())

  #def get_metric_result(cv_results):
  #results = pd.DataFrame.from_dict(cv_results)
  # results = results[results['status'] == "Success"]
  # cols = ['rank_test_scores', 'param_classifier:__choice__', 'mean_test_score']
  # cols.extend([key for key in cv_results.keys() if key.startswith('metric_')])
  # return results[cols].sort_values(by='rank_test_scores')

  #print("#" * 80)
  #print("metric results")
  #print(get_metric_result(pipeline_auto.cv_results_).to_string(index=False))

  return pipeline_auto



def do_it(df, config): #target_col, text_col):
    max_features  = config["max_features"]
    ngram_ranges  = config["ngram_ranges"]
    sublinear_tfs = config["sublinear_tfs"]
    stopwords     = config["stopwords"]
    n_components  = config["n_components"]

    res = []

    for x in itertools.product(max_features, ngram_ranges, sublinear_tfs, stopwords, n_components):
      logging.info("#" * 80)
      logging.info("#" * 80)
      logging.info("Combo: {}".format(x))
      logging.info("Building features...")
      features = get_text_features(df, 'en_clean',
                                   max_features = x[0],
                                   ngram_range = x[1],
                                   sublinear_tf = x[2],
                                   stop_words = x[3],
                                   n_components = x[4],
                                   get_lexical=True)
      logging.info("Done: shape={}".format(features.shape))

      logging.info("Running autosklearn...")
      #pipe = get_auto(X=features, y=df[target_col])
      #res.append({'combo': x, 'pipeline': pipe})


def main():

    # Read settings
    with open('settings_small.json') as f:
        config=json.load(f)

    logging.info("Read settings:")
    logging.info(json.dumps(config, indent=4, sort_keys=True))


    logging.info("Reading data file...")
    #df = pd.read_csv("https://drive.google.com/uc?export=download&id=1dzzVbgHphbCf7kvq9IKiIhwzmxPbuH4s")
    #df = df.sample(frac=0.02, replace=False, random_state=3)
    #df.to_csv("small.csv", index=False)
    df = pd.read_csv("small.csv")
    logging.info("Done. Shape = {}".format(df.shape))

    # Define our Target
    target_col = 'defaulted'
    text_feature = 'en_clean'

    df[target_col].value_counts()

    # Drop columns that don't make sense: leakage, IDs, etc.
    drop_list=(['loan_id'])
    df = df.drop(drop_list, axis=1)

    print(df.info())
    do_it(df, config) #target_col, text_feature)


if __name__ == "__main__":
  main()


