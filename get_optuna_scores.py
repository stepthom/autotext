import optuna
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--optuna-study-name', type=str, default='eq_lgbm_07')
    parser.add_argument('-s', '--optuna-storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    args = parser.parse_args()
    
    study = optuna.load_study( study_name=args.optuna_study_name, storage=args.optuna_storage)
    df = study.trials_dataframe(attrs=("number", "value", "duration", "params", "state"))

    print(args.optuna_study_name)
    print(df.shape)
    print(df['state'].value_counts())
    print(df.sort_values('value', ascending=False).head(20))
    
    #for trial in study.trials:
        #print(trial.user_attrs)
