import argparse
import numpy as np
import pandas as pd
import os

import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def dump_json(fn, json_obj):
    # Write json_obj to a file named fn

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

    print("DEBUG: Writing json file {}".format(fn))
    with open(fn, 'w') as fp:
        json.dump(json_obj, fp, indent=4, cls=NumpyEncoder)

def summarize(dir, output_file):

    # Read directory
    run_files = []
    for file in os.listdir(dir):
        if file.startswith('tune_') and file.endswith(".json"):
            run_files.append(os.path.join(dir, file))

    print("DEBUG: Found {} tune runs in {}.".format(len(run_files), dir))

    res = []
    from json.decoder import JSONDecodeError
    for run_file in run_files:
        run = {}
        with open(run_file) as f:

            try:
                run = json.load(f)
            except JSONDecodeError as e:
                print("ERROR: cannot parse json file {}".format(run_file))
                print(e)
                continue

            args = run.get('args', {})
            automl_settings = run.get('automl_settings', {})

            res.append({
                'run_file': run_file,
                'runname': run.get('runname', ''),
                #'data_id': data_id,
                'preds_fn': run.get('preds_fn', ''),
                'probas_fn': run.get('probas_fn', ''),
                #'config_summary': config_summary,
                #'num_indicator': data_sheet.get('num_indicator', ''),
                #'num_imputer': data_sheet.get('num_imputer', ''),
                #'cat_encoder': data_sheet.get('cat_encoder', ''),
                #'keep_top': data_sheet.get('keep_top', ''),
                #'special_imput_cols': str(data_sheet.get('special_impute_cols', '')),
                #'impute_cat': data_sheet.get('impute_cat', ''),
                #'autofeat': data_sheet.get('autofeat', ''),
                #'dimreduc': data_sheet.get('dimreduc', ''),
                #'feature_selector': data_sheet.get('feature_selector', ''),
                #'drop_cols': data_sheet.get('drop_cols', ''),
                #'custom_begin_funcs': data_sheet.get('custom_begin_funcs', ''),
                #'custom_end_funcs': data_sheet.get('custom_end_funcs', ''),
                #'min_sample_leaf': run.get('os_steve_min_sample_leaf', ''),
                #'smoothing': run.get('os_steve_smoothing', ''),
                #'geo_id_set': args.get('geo_id_set', ''),
                #'keep_top_id': args.get('keep_top_id', ''),
                'endtime': run.get('endtime', ''),
                'algo_set': run.get('algo_set', ''),
                #'search_type': automl_settings.get('search_type', ''),
                #'time_budget': automl_settings.get('time_budget', ''),
                'best_loss': run.get('best_loss', ''),
            })

    df = pd.DataFrame(res)
    if df.shape[0] > 0:
        df = df.sort_values('best_loss', ascending=True)
        #df = df.drop_duplicates(subset = ['data_id', 'search_type', 'search_time', 'eval_metric', 'ensemble', 'best_estimator', 'val_score'])
        print(df.shape)
        print(df[['runname', 'endtime', 'algo_set', 'best_loss']].head(20))
        df.to_csv(output_file, index=False)
        print("DEBUG: Wrote results file: {}".format(output_file))
    return


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    #summarize("pump/out", "pump/tune_results.csv")
    #summarize("h1n1/out", "h1n1/tune_results.csv")
    #summarize("seasonal/out", "seasonal/tune_results.csv")
    summarize("earthquake/out", "earthquake/tune_results.csv")


if __name__ == "__main__":
    main()
