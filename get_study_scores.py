import optuna
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--optuna-storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    args = parser.parse_args()
    
    study_names = ['h1n1', 'seasonal', 'eq']
    
    for study_name in study_names:
        print("===================================")
        print("Study: {}".format(study_name))

        study = optuna.load_study( study_name=study_name, storage=args.optuna_storage)
        df = study.trials_dataframe(attrs=("number", "value", "duration", "params", "state"))
        if len(df) > 0:
            print(df['state'].value_counts())
            print(df.sort_values('value', ascending=False).head(5))
       
    
