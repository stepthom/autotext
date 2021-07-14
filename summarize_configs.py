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

def summarize(data_dir, output_file):

    # Read directory
    data_sheet_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            data_sheet_files.append(os.path.join(data_dir, file))

    print("DEBUG: Found {} data sheets in {}.".format(len(data_sheet_files), data_dir))

    res = []
    for data_sheet_file in data_sheet_files:
        data_sheet = {}
        with open(data_sheet_file) as f:
            
            try:
                data_sheet = json.load(f)
            except:
                print("ERROR: cannot parse json file {}".format(data_sheet_file))
                continue

            data_id = data_sheet.get('data_id', None)
            config_summary = data_sheet.get('config_summary', None)
            if data_id is None or config_summary is None:
                print("DEBUG: skipping file is not a datasheet: {}".format(data_sheet_file))
                continue

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
            })

    df = pd.DataFrame(res)
    if df.shape[0] > 0:
        print(df.head())
        print(df.shape)
        df.to_csv(output_file, index=False)
        print("DEBUG: Wrote results file: {}".format(output_file))
    return


def main():    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    summarize("pump/data", "pump/data_sheets.csv")
    summarize("h1n1/data", "h1n1/data_sheets.csv")
    summarize("seasonal/data", "seasonal/data_sheets.csv")
    summarize("earthquake/data", "earthquake/data_sheets.csv")
   

if __name__ == "__main__":
    main()
