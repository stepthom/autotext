import optuna
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    parser.add_argument('-c', '--create-studies', type=int, default=0)
    parser.add_argument('-e', '--enqueue-trials', type=int, default=0)
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
                n_startup_trials = 100,
                multivariate = True,
                n_ei_candidates = 10,
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
            
            
    if args.enqueue_trials == 1:
        
        study = optuna.load_study(study_name="eq", storage=args.storage)
        
        # Some trials with known good settings in the past:
        # 739
        study.enqueue_trial({
              "learning_rate": 0.011098032,
              "min_child_weight": 0.131305556,
              "min_child_samples": 34,
              "feature_fraction_bynode": 0.229734382,
              "colsample_bytree": 0.817962118,
              "subsample": 0.893532361,
              "reg_alpha": 1.670653858,
              "reg_lambda": 0.035167253,
              "path_smooth": 0.0001,
              "cat_smooth": 5.979935576,
              "cat_l2": 22.86954873,
              "min_data_per_group": 189,
            
              #"max_bin": 39,
              #"num_leaves": 39,
        })
        
        # 742
        study.enqueue_trial({
              "learning_rate": 0.0114062972462413,
              "min_child_weight": 0.131305556,
              "min_child_samples": 59,
              "feature_fraction_bynode": 0.193417711,
              "colsample_bytree": 0.811681003450353,
              "subsample": 0.897379966897912,
              "reg_alpha": 1.50789227478525,
              "reg_lambda": 0.130148235, 
              "path_smooth": 0.0001,
              "cat_smooth": 5.91970281251434,
              "cat_l2": 23.790256777052,
              "min_data_per_group": 189,
            
              #"max_bin": 53,
              #"num_leaves": 40,
        })
        
        # 3246
        study.enqueue_trial({
           "learning_rate":0.00894104527254753,
           "min_child_weight":0.45829628488743057,
           "min_child_samples":8,
           "feature_fraction_bynode":0.24849772378500812,
           "colsample_bytree":0.734777331141205,
           "subsample":0.9994275346502477,
           "reg_alpha":0.009366546891440319,
           "reg_lambda":0.15152538835474738,
           "path_smooth":0.737410142067473,
           "cat_smooth":6.03074454863995,
           "cat_l2":5.455004586240047,
           "min_data_per_group":211,
            
           #"subsample_freq":1,
           #"max_bin":127,
           #"max_depth":31,
           #"num_leaves":127,
        })
            
        
        # Some ones I want to try manually
        study.enqueue_trial({
              "learning_rate": 0.015,
              "min_child_weight": 0.15,
              "min_child_samples": 59,
              "feature_fraction_bynode": 0.2,
              "colsample_bytree": 0.80,
              "subsample": 0.9,
              "reg_alpha": 1.5,
              "reg_lambda": 0.15,
              "path_smooth": 0.0001,
              "cat_smooth": 6.0,
              "cat_l2": 24.0,
              "min_data_per_group": 190,
            
              #"max_bin": 53,
              #"num_leaves": 40,
        })
        
        study.enqueue_trial({
              "learning_rate": 0.015,
              "min_child_weight": 0.15,
              "min_child_samples": 59,
              "feature_fraction_bynode": 0.9,
              "colsample_bytree": 0.80,
              "subsample": 0.9,
              "reg_alpha": 5.0,
              "reg_lambda": 5.0,
              "path_smooth": 0.1,
              "cat_smooth": 6.0,
              "cat_l2": 25.0,
              "min_data_per_group": 200,
            
              #"max_bin": 50,
              #"num_leaves": 40,
        })
        
        study.enqueue_trial({
              "learning_rate": 0.05,
              "min_child_weight": 0.15,
              "min_child_samples": 25,
              "feature_fraction_bynode": 0.9,
              "colsample_bytree": 0.9,
              "subsample": 0.9,
              "reg_alpha": 0.01,
              "reg_lambda": 0.01,
              "path_smooth": 0.01,
              "cat_smooth": 6.0,
              "cat_l2": 10.0,
              "min_data_per_group": 100,
            
              #"max_bin": 50,
              #"num_leaves": 40,
        })