import optuna
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-o', '--optuna-study-name', type=str, default='eq_lgbm_07')
    parser.add_argument('-s', '--optuna-storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    parser.add_argument('-d', '--delete-studies', type=int, default=0)
    args = parser.parse_args()
    
    #projects = ['eg', 'h1n1', 'seasonal', 'pump']
    #estimators = ['lgbm']
    study_name = "h1n1"
  
    print("===================================")
    print("Study: {}".format(study_name))

    if args.delete_studies == 1:
        print("Deleting study.")
        optuna.delete_study( study_name=study_name, storage=args.optuna_storage)

    else:
        study = optuna.load_study( study_name=study_name, storage=args.optuna_storage)
        print(study)
        print(study.user_attrs)
        df = study.trials_dataframe(attrs=("number", "value", "duration", "params", "state"))
        if len(df) > 0:
            print(df['state'].value_counts())
            print(df.sort_values('value', ascending=False).head(5))
       
    
