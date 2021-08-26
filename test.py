import optuna


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    
    study = optuna.load_study(
        study_name="lgbm_07", 
        storage="sqlite:///eq_studies.db", 
        )
    df = study.trials_dataframe(attrs=("number", "value", "duration", "params", "state"))
    
    print(df.sort_values('value', ascending=False).head(30))
    
    
    
    #study = optuna.load_study(
        #study_name="distributed-example", storage="postgresql://utkborfs:BR5DVp6qD9-Pe-JkkXKmayoWUB74fmdh@chunee.db.elephantsql.com/utkborfs"
    #)
    #study.optimize(objective, n_trials=100)