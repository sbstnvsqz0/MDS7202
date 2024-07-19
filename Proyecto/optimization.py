import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pandas as pd
from utils import rename_features_after_imputer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler
import os


SEED = 40

def get_args():
    parser = argparse.ArgumentParser("Optimizer")
    parser.add_argument('--name','-n',type=str,default=None,help='Abreviacion de modelo a optimizar, puede ser rfc, lgbm o xgb')
    parser.add_argument('--data','-d',type=str,default=None,help='Ruta del archivo de data')
    parser.add_argument('--labels','-l',type=str,default=None,help='Ruta de labels')
    parser.add_argument('--trials','-t',type=int,default=100,help='Número de trials que hace la optimización')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    assert args.name in ["rfc","lgbm","xgb"], "El nombre insertado debe estar entre rfc, lgbm y xgb"
    try:
        X = pd.read_csv(args.data)
    except ValueError: 
        print("Ruta de data es incorrecta")
    try:
        y = pd.read_csv(args.labels)
    except ValueError: 
        print("Ruta de labels es incorrecta")

    X_train,X_val,y_train,y_val = train_test_split(X,y.is_mob,test_size=0.3,random_state=SEED,stratify=y.is_mob)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info(f'Optimización de modelo {args.name}')

    robust_scaler_features = "AliasMatch,NewCribMonths,customer_age,intended_balcon_amount,BankSpots8w,DOBEmails4w,BankMonths,CreditCap,DaysSinceJob,ZipHustle,Speed6h,Speed24h,RiskScore,HustleMinutes,Speed4w"
    botar = "OldHoodMonths,DeviceScams"
    imputer_features = "BankMonths,HustleMinutes,DeviceEmails8w,RiskScore,NewCribMonths"
    one_hot_features = "income,JobStatus,CribStatus,LootMethod,InfoSource,DeviceOS,HustleMonth,DeviceEmails8w"
    pass_features = "AliveSession,CellPhoneCheck,ExtraPlastic,ForeignHustle,FreeMail,HomePhoneCheck"


    ####################################### RFC ##################################
    if args.name=="rfc":
        def objetive_function(trial):
            params_model = {"n_estimators": trial.suggest_int("n_estimators",30,1000),
                            "max_depth": trial.suggest_int("max_depth",5,50),
                            "min_samples_split": trial.suggest_int("min_samples_split",2,5)}

            params_encoder = {"drop":"first",
                              "sparse_output":False,
                              "handle_unknown":"ignore",
                            "min_frequency":trial.suggest_float("min_frequency",0,1)}
            
            
            params_imputer = {"strategy": trial.suggest_categorical("strategy",["mean","median","most_frequent"])}

            imputer = ColumnTransformer([("imputer",SimpleImputer(missing_values=-1,**params_imputer),imputer_features.split(","))],
                                        remainder="passthrough")
            imputer.set_output(transform='pandas')

            col_transformer = ColumnTransformer([("encoder",OneHotEncoder(**params_encoder),rename_features_after_imputer(imputer_features,one_hot_features).split(",")),
                                                ("RobustScaler",RobustScaler(),rename_features_after_imputer(imputer_features,robust_scaler_features).split(",")),
                                                ("passthrough","passthrough",rename_features_after_imputer(imputer_features,pass_features).split(","))],
                                                remainder = "drop")
            col_transformer.set_output(transform='pandas')

            pipeline = Pipeline([("imputer",imputer),
                                ("col_transformer",col_transformer),
                                ("clf",RandomForestClassifier(class_weight = "balanced",random_state=SEED,**params_model))])

            pipeline.fit(X_train,y_train)
            y_pred = pipeline.predict(X_val)
            return f1_score(y_val,y_pred)
    
    ####################################### LGBM ##################################
    
    elif args.name == "lgbm":
        def objective_function(trial):
            params_model = {"num_leaves": trial.suggest_int("num_leaves",30,120),
                            "max_depth": trial.suggest_int("max_depth",3,10),
                            "learning_rate": trial.suggest_float("learning_rate",0.001,0.1,log=True),
                            "n_estimators": trial.suggest_int("n_estimators ",20,100),
                            "min_child_samples":trial.suggest_int("min_child_samples",5,10),
                            "reg_alpha": trial.suggest_float("reg_alpha",0,1),
                            "reg_lambda": trial.suggest_float("reg_lambda",0,1)}

            params_encoder = {"drop":"first",
                              "sparse_output":False,
                              "handle_unknown":"ignore",
                            "min_frequency":trial.suggest_float("min_frequency",0,1)}
            
            params_imputer = {"strategy": trial.suggest_categorical("strategy",["mean","median","most_frequent"])}

            imputer = ColumnTransformer([("imputer",SimpleImputer(missing_values=-1,**params_imputer),imputer_features.split(","))],
                                        remainder="passthrough")
            imputer.set_output(transform='pandas')

            col_transformer = ColumnTransformer([("encoder",OneHotEncoder(**params_encoder),rename_features_after_imputer(imputer_features,one_hot_features).split(",")),
                                                ("RobustScaler",RobustScaler(),rename_features_after_imputer(imputer_features,robust_scaler_features).split(",")),
                                                ("passthrough","passthrough",rename_features_after_imputer(imputer_features,pass_features).split(","))],
                                                remainder = "drop")
            col_transformer.set_output(transform='pandas')

            pipeline = Pipeline([("imputer",imputer),
                                ("col_transformer",col_transformer),
                                ("clf",LGBMClassifier(verbose=-1,class_weight= "balanced",random_state=SEED,**params_model))])
            
            pipeline.fit(X_train,y_train)
            y_pred = pipeline.predict(X_val)
            return f1_score(y_val,y_pred)
    
    ####################################### XGB ##################################

    elif args.name == "xgb":
        def objective_function(trial):
            params_model = {"learning_rate": trial.suggest_float("learning_rate",0.01,0.1,log=True),
                            "n_estimators": trial.suggest_int("n_estimators",20,1000),
                            "max_depth": trial.suggest_int("max_depth",20,100),
                            "max_leaves": trial.suggest_int("max_leaves",5,20),
                            "min_child_weight": trial.suggest_int("min_child_weight",1,5),
                            "reg_alpha": trial.suggest_float("reg_alpha",0,1),
                            "reg_lambda": trial.suggest_float("reg_lambda",0,1)}
            
            params_encoder = {"drop":"first",
                              "sparse_output":False,
                              "handle_unknown":"ignore",
                            "min_frequency":trial.suggest_float("min_frequency",0,1)}
            
            params_imputer = {"strategy": trial.suggest_categorical("strategy",["mean","median","most_frequent"])}

            imputer = ColumnTransformer([("imputer",SimpleImputer(missing_values=-1,**params_imputer),imputer_features.split(","))],
                                        remainder="passthrough")
            imputer.set_output(transform='pandas')

            col_transformer = ColumnTransformer([("encoder",OneHotEncoder(**params_encoder),rename_features_after_imputer(imputer_features,one_hot_features).split(",")),
                                                ("RobustScaler",RobustScaler(),rename_features_after_imputer(imputer_features,robust_scaler_features).split(",")),
                                                ("passthrough","passthrough",rename_features_after_imputer(imputer_features,pass_features).split(","))],
                                                remainder = "drop")
            col_transformer.set_output(transform='pandas')

            pipeline = Pipeline([("imputer",imputer),
                                ("col_transformer",col_transformer),
                                ("clf",XGBClassifier(scale_pos_weight=100,random_state=SEED,**params_model))])
            
            pipeline.fit(X_train,y_train)
            y_pred = pipeline.predict(X_val)
            return f1_score(y_val,y_pred)
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",sampler=TPESampler())
    study.optimize(objective_function,n_trials=args.trials,show_progress_bar=True)

    t = args.data.split("_")[1]

    with open(f"optim_results/results_{args.name}_{t}.csv","a") as f:
            f.write("{best_value},{best_params}".format(best_value=study.best_value,best_params=study.best_params))
            f.write("\n")
            f.close()
    logging.info(f'Optimización de modelo {args.name} finalizada')


