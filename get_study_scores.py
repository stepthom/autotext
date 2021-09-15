import optuna
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--optuna-storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    parser.add_argument('-s', '--save-results', type=int, default=0)
    args = parser.parse_args()

    #study_names = ['h1n1', 'seasonal', 'eq']
    study_names = ['eq']

    for study_name in study_names:
        print("===================================")
        print("Study: {}".format(study_name))

        study = optuna.load_study( study_name=study_name, storage=args.optuna_storage)
        
        df = study.trials_dataframe()
        if len(df) > 0:
            print(df['state'].value_counts())
            print("Most recent:")
            print(df[df.state == "COMPLETE"].sort_values('number', ascending=False).head(3))
            print("Best:")
            df = df.sort_values('value', ascending=False)
            print(df.head(15))
            if args.save_results == 1:
                df.to_csv('out/{}.csv'.format(study_name), index=False)

            imp = optuna.importance.get_param_importances(study)
            print("Hyperparam importances:")
            print(imp)

            iss = optuna.samplers.IntersectionSearchSpace()
            print("Intersection Search Spaces:")
            print(iss.calculate(study))


