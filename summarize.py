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
        if file.startswith('run') and file.endswith(".json"):
            run_files.append(os.path.join(dir, file))

    print("DEBUG: Found {} runs in {}.".format(len(run_files), dir))

    res = []
    for run_file in run_files:
        run = {}
        with open(run_file) as f:
            
            try:
                run = json.load(f)
            except:
                print("ERROR: cannot parse json file {}".format(run_file))
                continue

            data_id = run.get('data_id', None)
            data_sheet = run.get('data_sheet', None)
            if data_id is None or data_sheet is None:
                print("DEBUG: skipping file is not a datasheet: {}".format(data_sheet_file))
                continue

            config_summary = data_sheet.get('config_summary', None)
            comp_settings = run.get('comp_settings', None)

            res.append({
                'data_id': data_id,
                'config_summary': config_summary,
                'num_indicator': data_sheet.get('num_indicator', ''),
                'num_imputer': data_sheet.get('num_imputer', ''),
                'cat_encoder': data_sheet.get('cat_encoder', ''),
                'keep_top': data_sheet.get('keep_top', ''),
                'special_imput_cols': str(data_sheet.get('special_impute_cols', '')),
                'impute_cat': data_sheet.get('impute_cat', ''),
                'autofeat': data_sheet.get('autofeat', ''),
                'dimreduc': data_sheet.get('dimreduc', ''),
                'feature_selector': data_sheet.get('feature_selector', ''),
                'drop_cols': data_sheet.get('drop_cols', ''),
                'custom_begin_funcs': data_sheet.get('custom_begin_funcs', ''),
                'custom_end_funcs': data_sheet.get('custom_end_funcs', ''),
                'runname': run.get('runname', ''),
                'endtime': run.get('endtime', ''),
                'search_type': comp_settings.get('search_type', ''),
                'search_time': comp_settings.get('search_time', ''),
                'eval_metric': comp_settings.get('eval_metric', ''),
                'ensemble': comp_settings.get('ensemble', ''),
                'val_score': run.get('val_score', ''),
            })

    df = pd.DataFrame(res)
    if df.shape[0] > 0:
        df = df.sort_values('val_score', ascending=False)
        print(df.head())
        print(df.shape)
        df.to_csv(output_file, index=False)
        print("DEBUG: Wrote results file: {}".format(output_file))
    return


def main():    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    summarize("pump/out", "pump/results.csv")
    summarize("h1n1/out", "h1n1/results.csv")
    summarize("seasonal/out", "seasonal/results.csv")
    summarize("earthquake/out", "earthquake/results.csv")
   

if __name__ == "__main__":
    main()
