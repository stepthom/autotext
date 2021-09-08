import optuna
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    parser.add_argument('-c', '--create-studies', type=int, default=0)
    parser.add_argument('-d', '--delete-studies', type=int, default=0)
    args = parser.parse_args()
   
    if args.create_studies == 1:
    
        study = optuna.create_study(
            study_name="seasonal",
            storage=args.storage,
            sampler= optuna.samplers.TPESampler(
                n_startup_trials = 100,
                n_ei_candidates = 10,
                constant_liar=True,
            ),
            direction="maximize",
            load_if_exists = True,
        )
        study.set_user_attr("id_col",  "respondent_id")
        study.set_user_attr("target_col",  "seasonal_vaccine")
        study.set_user_attr("out_dir",  "seasonal/out")
        study.set_user_attr("train_fn", "seasonal/vaccine_seasonal_train.csv")
        study.set_user_attr("test_fn", "seasonal/vaccine_seasonal_test.csv")
        study.set_user_attr("metric", "roc_auc")


        study = optuna.create_study(
            study_name="h1n1",
            storage=args.storage,
            sampler= optuna.samplers.TPESampler(
                n_startup_trials = 100,
                n_ei_candidates = 10,
                constant_liar=True,
            ),
            direction="maximize",
            load_if_exists = True,
        )
        study.set_user_attr("id_col",  "respondent_id")
        study.set_user_attr("target_col",  "h1n1_vaccine")
        study.set_user_attr("out_dir",  "h1n1/out")
        study.set_user_attr("train_fn", "h1n1/vaccine_h1n1_train.csv")
        study.set_user_attr("test_fn", "h1n1/vaccine_h1n1_test.csv")
        study.set_user_attr("metric", "roc_auc")


        study = optuna.create_study(
            study_name="eq",
            storage=args.storage,
            sampler= optuna.samplers.TPESampler(
                n_startup_trials = 400,
                n_ei_candidates = 20,
                constant_liar=True,
            ),
            direction="maximize",
            load_if_exists = True,
        )
        study.set_user_attr("id_col",  "building_id")
        study.set_user_attr("target_col",  "damage_grade")
        study.set_user_attr("out_dir",  "earthquake/out")
        study.set_user_attr("train_fn", "earthquake/earthquake_train.csv")
        study.set_user_attr("test_fn", "earthquake/earthquake_test.csv")
        study.set_user_attr("metric", "f1")

        study = optuna.create_study(
            study_name="pump",
            storage=args.storage,
            sampler= optuna.samplers.TPESampler(
                n_startup_trials = 100,
                n_ei_candidates = 10,
                constant_liar=True,
            ),
            direction="maximize",
            load_if_exists = True,
        )
        study.set_user_attr("id_col",  "id")
        study.set_user_attr("target_col",  "status_group")
        study.set_user_attr("out_dir",  "pump/out")
        study.set_user_attr("train_fn", "pump/pump_train.csv")
        study.set_user_attr("test_fn", "pump/pump_test.csv")
        study.set_user_attr("metric", "accuracy")
        
    if args.delete_studies == 1:
        #study_names = ['h1n1', 'seasonal', 'eq', 'pump']
        study_names = ['eq']
        for study_name in study_names:
            print("Deleting study: {}".format(study_name))
            optuna.delete_study(study_name=study_name, storage=args.storage)