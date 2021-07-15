import uuid
import argparse
import pandas as pd
import os
import numpy as np

import json
import datetime

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
        
    with open(fn, 'w') as fp:
        json.dump(json_obj, fp, indent=4, cls=NumpyEncoder)


def main():    
    parser = argparse.ArgumentParser()
    #parser.add_argument('-s', '--run-settings', default='default_settings.json')
    args = parser.parse_args()
    
    def combine_and_write(probas_fns, out_dir, id_col, target_col, classes, make_preds=True):
        
        runname = str(uuid.uuid4())

        combined = {}
        combined['runname'] = runname
        combined['startime'] = str(datetime.datetime.now())
        combined['probas_fns'] = probas_fns
        combined['out_dir'] = out_dir

        print("Combining files: {}".format(probas_fns))
       
        # Read one of the files to get the IDs
        combined_df = pd.read_csv(probas_fns[0], usecols=[id_col])
        
        probas = list()
        for probas_fn in probas_fns:
            _df = pd.read_csv(probas_fn)
            probas.append(_df.drop(id_col, axis=1))
        
        mean_df = pd.DataFrame(np.mean(probas, axis=0), columns=classes)
        
        combined_df = pd.concat([combined_df, mean_df], axis=1)
        
      
        if make_preds:
            combined_df[target_col] = combined_df[classes].idxmax(axis=1)
            print(combined_df[target_col].value_counts())
            
        else:
            # leave probas
            combined_df[target_col] = combined_df[classes[-1]]
        
        print(combined_df.head())
        
        out_fn = os.path.join(out_dir, 'combined-{}-preds.csv'.format(runname))
        combined['out_fn'] = out_fn
        
        combined_df[[id_col, target_col]].to_csv(out_fn, index=False)
        print("Wrote probas file {}".format(out_fn))
        
        json_fn = os.path.join(out_dir, "combined-{}.json".format(runname))
        dump_json(json_fn, combined)
        
        return out_fn, runname
    
    ##########################
    # H1N1
    ##########################
    
    # One way is to just get the top N results from the results file
    num_top = 5
    results = pd.read_csv("h1n1/results.csv").head(num_top)
    probas_fns = list(results['probas_fn'])
    
    # Another way is to manually list them 
    #probas_fns = []
   
    h1n1_out_fn, h1n1_runname = combine_and_write(
        probas_fns,
        "h1n1/out", 
        "respondent_id", 
        "h1n1_vaccine", 
        classes= ["0", "1"],
        make_preds = False
    )
    
    ##########################
    # Seasonal
    ##########################
    
    # One way is to just get the top N results from the results file
    num_top = 5
    results = pd.read_csv("seasonal/results.csv").head(num_top)
    probas_fns = list(results['probas_fn'])
    
    # Another way is to manually list them 
    #probas_fns = []
   
    seasonal_out_fn, seasonal_runname = combine_and_write(
        probas_fns,
        "seasonal/out", 
        "respondent_id", 
        "seasonal_vaccine", 
        classes= ["0", "1"],
        make_preds = False
    )
    
    #################################
    # Now combine H1N1 and Seasonal
    #################################

    h1n1_df = pd.read_csv(h1n1_out_fn)
    seasonal_df = pd.read_csv(seasonal_out_fn)

    assert(h1n1_df['respondent_id'].equals(seasonal_df['respondent_id']))

    vaccine_df = pd.DataFrame(
        {'respondent_id': h1n1_df['respondent_id'], 
         'h1n1_vaccine': h1n1_df['h1n1_vaccine'], 
         'seasonal_vaccine': seasonal_df['seasonal_vaccine']
        })
    
    print(vaccine_df.head())
    
    new_fn = "{}_{}_vaccine-combined-preds.csv".format(h1n1_runname, seasonal_runname)
    print("Writing vaccine combined filename: {}".format(new_fn))
    vaccine_df.to_csv("vaccine/out/{}".format(new_fn), index=False)
    
    
    
    ##########################
    # Pump
    ##########################
    
    # One way is to just get the top N results from the results file
    num_top = 5
    results = pd.read_csv("pump/results.csv").head(num_top)
    probas_fns = list(results['probas_fn'])
    
    # Another way is to manually list them 
    #probas_fns = []
   
    _, _ =  combine_and_write(
        probas_fns,
        "pump/out", 
        "id", 
        "status_group", 
        classes= ['functional', 'functional needs repair', 'non functional'],
        make_preds = True
    )
    
    ##########################
    # Earthquake
    ##########################
    
    # One way is to just get the top N results from the results file
    num_top = 3
    results = pd.read_csv("earthquake/results.csv").head(num_top)
    probas_fns = list(results['probas_fn'])
    
    # Another way is to manually list them 
    #probas_fns = []
   
    _, _ = combine_and_write(
        probas_fns,
        "earthquake/out", 
        "building_id", 
        "damage_grade", 
        classes= ["1", "2", "3"],
        make_preds = True
    )
    
    

if __name__ == "__main__":
    main()
