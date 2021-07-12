import argparse
import numpy as np
import pandas as pd
import os

import json

import jsonpickle

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


def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--competition', default='pump')
    args = parser.parse_args()
   
    data_dir = None
    output_file = None
    if args.competition.startswith("p"):
        data_dir = "pump/data"
        output_file = "pump/results.csv"
    elif args.competition.startswith("h"):
        data_dir = "h1n1/data"
        output_file = "h1n1/results.csv"
    elif args.competition.startswith("s"):
        data_dir = "seasonal/data"
        output_file = "seasonal/results.csv"
    elif args.competition.startswith("e"):
        data_dir = "earthquake/data"
        output_file = "earthquake/results.csv"
    else:
        print("Error: unknown competition type: {}".format(args.competition))
        return
    
    
    # Read directory
    data_sheet_files = []
    for file in os.listdir(data_dir):
        if file.endswith(".json"):
            data_sheet_files.append(os.path.join(data_dir, file))

    print("DEBUG: Found {} data sheets.".format(len(data_sheet_files)))
   
    res = []
    for data_sheet_file in data_sheet_files:
        data_sheet = {}
        with open(data_sheet_file) as f:
            data_sheet = json.load(f)
            
            data_id = data_sheet.get('data_id', None)
            config_summary = data_sheet.get('config_summary', None)
            if data_id is None or config_summary is None:
                print("DEBUG: skipping file is not a datasheet: {}".format(data_sheet_file))
                continue
                
            runs = data_sheet.get('runs', {})
            for run in runs:
                
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
                    'runname': run,
                    'endtime': runs[run].get('endtime', ''),
                    'search_type': runs[run].get('search_type', ''),
                    'search_time': runs[run].get('search_time', ''),
                    'eval_metric': runs[run].get('eval_metric', ''),
                    'val_score': runs[run].get('val_score', ''),
                })
                
    df = pd.DataFrame(res)
    df = df.sort_values('val_score', ascending=False)
    print(df)
    df.to_csv(output_file, index=False)
    print("DEBUG: Wrote results file: {}".format(output_file))

if __name__ == "__main__":
    main()
