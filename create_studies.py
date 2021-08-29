import optuna
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('-o', '--optuna-study-name', type=str, default='eq_lgbm_07')
    parser.add_argument('-s', '--storage', type=str, default="postgresql://hpc3552@172.20.13.14/hpc3552")
    #parser.add_argument('-d', '--delete-studies', type=int, default=0)
    args = parser.parse_args()
    
    
    study = optuna.create_study(
        study_name="h1n1",
        storage="postgresql://hpc3552@172.20.13.14/hpc3552",
        sampler= optuna.samplers.TPESampler(
            n_startup_trials = 100,
            n_ei_candidates = 10,
            constant_liar=True,
        ),
        direction="maximize",
        load_if_exists = True,
    )
    pipe_args = {}
    pipe_args['01'] = {
       "drop_cols":[

       ],
       "num_cols_impute":[
          "h1n1_concern",
          "h1n1_knowledge",
          "behavioral_antiviral_meds",
          "behavioral_avoidance",
          "behavioral_face_mask",
          "behavioral_wash_hands",
          "behavioral_large_gatherings",
          "behavioral_outside_home",
          "behavioral_touch_face",
          "doctor_recc_h1n1",
          "doctor_recc_seasonal",
          "chronic_med_condition",
          "child_under_6_months",
          "health_worker",
          "health_insurance",
          "opinion_h1n1_vacc_effective",
          "opinion_h1n1_risk",
          "opinion_h1n1_sick_from_vacc",
          "opinion_seas_vacc_effective",
          "opinion_seas_risk",
          "opinion_seas_sick_from_vacc",
          "household_adults",
          "household_children"
       ],
       "num_cols_indicator":[
          "h1n1_concern",
          "h1n1_knowledge",
          "behavioral_antiviral_meds",
          "behavioral_avoidance",
          "behavioral_face_mask",
          "behavioral_wash_hands",
          "behavioral_large_gatherings",
          "behavioral_outside_home",
          "behavioral_touch_face",
          "doctor_recc_h1n1",
          "doctor_recc_seasonal",
          "chronic_med_condition",
          "child_under_6_months",
          "health_worker",
          "health_insurance",
          "opinion_h1n1_vacc_effective",
          "opinion_h1n1_risk",
          "opinion_h1n1_sick_from_vacc",
          "opinion_seas_vacc_effective",
          "opinion_seas_risk",
          "opinion_seas_sick_from_vacc",
          "household_adults",
          "household_children"
       ],
       "cat_cols_impute":[
          "age_group",
          "education",
          "race",
          "sex",
          "income_poverty",
          "marital_status",
          "rent_or_own",
          "employment_status",
          "hhs_geo_region",
          "census_msa",
          "employment_industry",
          "employment_occupation"
       ],
       "cat_cols_smush":[
          "age_group",
          "education",
          "race",
          "sex",
          "income_poverty",
          "marital_status",
          "rent_or_own",
          "employment_status",
          "hhs_geo_region",
          "census_msa",
          "employment_industry",
          "employment_occupation"
       ],
       "cat_cols_onehot_encode":[

       ],
       "cat_cols_target_encode":[

       ],
       "cat_cols_ordinal_encode":[
          "age_group",
          "education",
          "race",
          "sex",
          "income_poverty",
          "marital_status",
          "rent_or_own",
          "employment_status",
          "hhs_geo_region",
          "census_msa",
          "employment_industry",
          "employment_occupation"
       ],
       "float_cols":[

       ],
       "autofeat":0,
       "normalize":0
    }

    pipe_args['02'] = {
       "drop_cols":[

       ],
       "num_cols_impute":[
          "h1n1_concern",
          "h1n1_knowledge",
          "behavioral_antiviral_meds",
          "behavioral_avoidance",
          "behavioral_face_mask",
          "behavioral_wash_hands",
          "behavioral_large_gatherings",
          "behavioral_outside_home",
          "behavioral_touch_face",
          "doctor_recc_h1n1",
          "doctor_recc_seasonal",
          "chronic_med_condition",
          "child_under_6_months",
          "health_worker",
          "health_insurance",
          "opinion_h1n1_vacc_effective",
          "opinion_h1n1_risk",
          "opinion_h1n1_sick_from_vacc",
          "opinion_seas_vacc_effective",
          "opinion_seas_risk",
          "opinion_seas_sick_from_vacc",
          "household_adults",
          "household_children"
       ],
       "num_cols_indicator":[
          "h1n1_concern",
          "h1n1_knowledge",
          "behavioral_antiviral_meds",
          "behavioral_avoidance",
          "behavioral_face_mask",
          "behavioral_wash_hands",
          "behavioral_large_gatherings",
          "behavioral_outside_home",
          "behavioral_touch_face",
          "doctor_recc_h1n1",
          "doctor_recc_seasonal",
          "chronic_med_condition",
          "child_under_6_months",
          "health_worker",
          "health_insurance",
          "opinion_h1n1_vacc_effective",
          "opinion_h1n1_risk",
          "opinion_h1n1_sick_from_vacc",
          "opinion_seas_vacc_effective",
          "opinion_seas_risk",
          "opinion_seas_sick_from_vacc",
          "household_adults",
          "household_children"
       ],
       "cat_cols_impute":[
          "age_group",
          "education",
          "race",
          "sex",
          "income_poverty",
          "marital_status",
          "rent_or_own",
          "employment_status",
          "hhs_geo_region",
          "census_msa",
          "employment_industry",
          "employment_occupation"
       ],
       "cat_cols_onehot_encode":[

       ],
       "cat_cols_target_encode":[
          "age_group",
          "education",
          "race",
          "sex",
          "income_poverty",
          "marital_status",
          "rent_or_own",
          "employment_status",
          "hhs_geo_region",
          "census_msa",
          "employment_industry",
          "employment_occupation"
       ],
       "cat_cols_ordinal_encode":[

       ],
       "float_cols":[

       ],
       "autofeat":0,
       "normalize":0
    }
    study.set_user_attr("pipe_args", pipe_args)
    study.set_user_attr("id_col",  "respondent_id")
    study.set_user_attr("target_col",  "h1n1_vaccine")
    study.set_user_attr("out_dir",  "h1n1/out")
    study.set_user_attr("train_fn", "h1n1/vaccine_h1n1_train.csv")
    study.set_user_attr("test_fn", "h1n1/vaccine_h1n1_test.csv")
        #objective = 'binary',
        #num_class = 1,
    study.set_user_attr("metric", "roc_auc")
    
    #projects = ['eg', 'h1n1', 'seasonal', 'pump']
    #estimators = ['lgbm']
  
    
