from comet_ml import Experiment
import argparse
import numpy as np
import pandas as pd
import os
import sys


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Comet params
    parser.add_argument('--enable-comet', type=int, default=1)
    # Prep/FE params

    args = parser.parse_args()

    print("Command line arguments:")
    print(vars(args))
    
    from comet_ml import API
    
    comet_api = API()
    
    exps = comet_api.get('stepthom', 'eq-searcher')
   
    res = []
    for exp in exps:
        name = exp.get_name()
        key = exp.get_metadata()['experimentKey']
        print(name)
        algo_set = exp.get_parameters_summary(parameter='algo_set')['valueCurrent']
        geo_id_set = exp.get_parameters_summary(parameter='geo_id_set')['valueCurrent']
        autofeat = exp.get_parameters_summary(parameter='autofeat')['valueCurrent']
        normalize = exp.get_parameters_summary(parameter='normalize')['valueCurrent']
        sampler = exp.get_parameters_summary(parameter='sampler')['valueCurrent']
        best_value = 0.0
        best_step = 0
        step = -1
        for metric in exp.get_metrics():
            if metric['metricName'] != 'mean_val_score':
                continue
            step = step + 1
            value = float(metric['metricValue'])
            if value > best_value:
                best_value = value
                best_step = step
            
        res.append({
            'name': name,
            'key': key,
            'algo_set': algo_set,
            'geo_id_set': geo_id_set,
            'autofeat': autofeat,
            'normalize': normalize,
            'sampler': sampler,
            'steps': step,
            'best_step': best_step,
            'best_value': best_value,
        }) 
        #print(best_value)
        
    df = pd.DataFrame(res)
    df  = df.sort_values('best_value', ascending=False)
    print(df.head(20))
    


    # Create an experiment with your api key
    #exp=None
    #if args.enable_comet == 1:
        #exp = Experiment(
            #project_name="eq_searcher",
            #workspace="stepthom",
        #)


if __name__ == "__main__":
    main()
