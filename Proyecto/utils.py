from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def exploratory_data_analysis(dataframe):
    # shape
    r,c = dataframe.shape
    print(f"El dataframe tiene {r} filas y {c} columnas\n")

    #nombre de columnas
    c_names = dataframe.columns.values
    print(f"El DataFrame esta compuesto por las siguientes columnas: {c_names}\n")

    #vista general
    first5 = dataframe[:5]
    print(f"Primeras 5 filas:\n ")
    display(first5)

    #info
    print("Info general de dataframe:\n")
    print(dataframe.info())

    # describe
    desc = dataframe.describe()
    print(f"Descripción numérica del Dataframe:\n")
    display(desc)

    # Valores nulos por columna
    cols_nulls = (dataframe==-1).sum()
    print(f"Cantidad de valores nulos por columna:\n")
    display(cols_nulls)

    #Valores únicos por columna
    uniq_x_cols = dataframe.nunique()
    print("Cantidad de valores únicos por columna:\n")
    display(uniq_x_cols)


def discrete_graph(df,cols):
    #4 por fila
    n_rows=len(cols)//4+ 1 if len(cols)%4!=0 else len(cols)/4
    fig = make_subplots(rows=n_rows,cols=4)
    for k in range(len(cols)):
        col,row = k%4+1,k//4+1
        fig.add_trace(go.Histogram(x=df[cols[k]],name=f"Histograma de {cols[k]}"),row=row,col=col)
        fig.update_layout(height=1200,width=1600,title_text="Histogramas de las variables discretas")
    fig.write_html("Histogramas.html")


def cont_graph(df,cols):
    fig = make_subplots(rows=len(cols)+1,cols=2)
    for k in range(len(cols)):
        fig.add_trace(go.Box(y=df[cols[k]],name=f"Boxplot de {cols[k]}"),row=k+1,col=1)
        fig.add_trace(go.Histogram(x=df[cols[k]],name=f"Histograma de {cols[k]}"),row=k+1,col=2)
        fig.update_layout(height=1200,width=1600,title_text="Gráficos de las variables continuas")
    fig.write_html("Histogramas_y_boxplot.html")


def show_corr(df):
    sns.heatmap(df.select_dtypes(include='number').corr().abs()>0.6)  # th de 0.6 para ver existencia
    plt.show()

def rename_features_after_imputer(imputer_features,other_features):
    list_features = other_features.split(",")
    list_imputer_features = imputer_features.split(",")
    new_list=[]
    for feature in list_features:
        if feature in list_imputer_features:
            new_list.append("imputer__"+feature)
        else:
            new_list.append("remainder__"+feature)
    return ",".join(new_list)

def eval_pipe(pipeline,X,y):
    y_hat = pipeline.predict(X)
    print(classification_report(y,y_hat))

def pipe_rfc(grid,seed=40):

    robust_scaler_features = "AliasMatch,NewCribMonths,customer_age,intended_balcon_amount,BankSpots8w,DOBEmails4w,BankMonths,CreditCap,DaysSinceJob,ZipHustle,Speed6h,Speed24h,RiskScore,HustleMinutes,Speed4w"
    botar = "OldHoodMonths,DeviceScams"
    imputer_features = "BankMonths,HustleMinutes,DeviceEmails8w,RiskScore,NewCribMonths"
    one_hot_features = "income,JobStatus,CribStatus,LootMethod,InfoSource,DeviceOS,HustleMonth,DeviceEmails8w"
    pass_features = "AliveSession,CellPhoneCheck,ExtraPlastic,ForeignHustle,FreeMail,HomePhoneCheck"

    imputer = ColumnTransformer([("imputer",SimpleImputer(missing_values=-1,strategy = grid["strategy"]),imputer_features.split(","))],
                                        remainder="passthrough")
    imputer.set_output(transform='pandas')

    col_transformer = ColumnTransformer([("encoder",OneHotEncoder(drop = "first",
                                                                    sparse_output = False,
                                                                    handle_unknown="ignore",
                                                                    min_frequency = grid["min_frequency"]),rename_features_after_imputer(imputer_features,one_hot_features).split(",")),
                                                ("RobustScaler",RobustScaler(),rename_features_after_imputer(imputer_features,robust_scaler_features).split(",")),
                                                ("passthrough","passthrough",rename_features_after_imputer(imputer_features,pass_features).split(","))],
                                                remainder = "drop")
    col_transformer.set_output(transform='pandas')

    pipeline = Pipeline([("imputer",imputer),
                        ("col_transformer",col_transformer),
                        ("clf",RandomForestClassifier(n_estimators = grid["n_estimators"],
                                                        max_depth  = grid["max_depth"],
                                                        min_samples_split = grid["min_samples_split"] ,
                                                        class_weight = "balanced",random_state=seed))])
    return pipeline


def pipe_xgb(grid,seed=40):

    robust_scaler_features = "AliasMatch,NewCribMonths,customer_age,intended_balcon_amount,BankSpots8w,DOBEmails4w,BankMonths,CreditCap,DaysSinceJob,ZipHustle,Speed6h,Speed24h,RiskScore,HustleMinutes,Speed4w"
    botar = "OldHoodMonths,DeviceScams"
    imputer_features = "BankMonths,HustleMinutes,DeviceEmails8w,RiskScore,NewCribMonths"
    one_hot_features = "income,JobStatus,CribStatus,LootMethod,InfoSource,DeviceOS,HustleMonth,DeviceEmails8w"
    pass_features = "AliveSession,CellPhoneCheck,ExtraPlastic,ForeignHustle,FreeMail,HomePhoneCheck"

    imputer = ColumnTransformer([("imputer",SimpleImputer(missing_values=-1,strategy = grid["strategy"]),imputer_features.split(","))],
                                        remainder="passthrough")
    imputer.set_output(transform='pandas')

    col_transformer = ColumnTransformer([("encoder",OneHotEncoder(drop = "first",
                                                                    sparse_output = False,
                                                                    handle_unknown="ignore",
                                                                    min_frequency = grid["min_frequency"]),rename_features_after_imputer(imputer_features,one_hot_features).split(",")),
                                                ("RobustScaler",RobustScaler(),rename_features_after_imputer(imputer_features,robust_scaler_features).split(",")),
                                                ("passthrough","passthrough",rename_features_after_imputer(imputer_features,pass_features).split(","))],
                                                remainder = "drop")
    col_transformer.set_output(transform='pandas')

    pipeline = Pipeline([("imputer",imputer),
                        ("col_transformer",col_transformer),
                        ("clf",XGBClassifier(learning_rate = grid["learning_rate"],
                                            n_estimators =  grid["n_estimators"],
                                            max_depth = grid["max_depth"],
                                            max_leaves = grid["max_leaves"],
                                            min_child_weight = grid["min_child_weight"],
                                            reg_alpha = grid["reg_alpha"],
                                            reg_lambda = grid["reg_lambda"],
                                            scale_pos_weight=100,
                                            random_state=seed))])
    return pipeline

def pipe_lgbm(grid,seed=40):

    robust_scaler_features = "AliasMatch,NewCribMonths,customer_age,intended_balcon_amount,BankSpots8w,DOBEmails4w,BankMonths,CreditCap,DaysSinceJob,ZipHustle,Speed6h,Speed24h,RiskScore,HustleMinutes,Speed4w"
    botar = "OldHoodMonths,DeviceScams"
    imputer_features = "BankMonths,HustleMinutes,DeviceEmails8w,RiskScore,NewCribMonths"
    one_hot_features = "income,JobStatus,CribStatus,LootMethod,InfoSource,DeviceOS,HustleMonth,DeviceEmails8w"
    pass_features = "AliveSession,CellPhoneCheck,ExtraPlastic,ForeignHustle,FreeMail,HomePhoneCheck"

    imputer = ColumnTransformer([("imputer",SimpleImputer(missing_values=-1,strategy = grid["strategy"]),imputer_features.split(","))],
                                        remainder="passthrough")
    imputer.set_output(transform='pandas')

    col_transformer = ColumnTransformer([("encoder",OneHotEncoder(drop = "first",
                                                                    sparse_output = False,
                                                                    handle_unknown="ignore",
                                                                    min_frequency = grid["min_frequency"]),rename_features_after_imputer(imputer_features,one_hot_features).split(",")),
                                                ("RobustScaler",RobustScaler(),rename_features_after_imputer(imputer_features,robust_scaler_features).split(",")),
                                                ("passthrough","passthrough",rename_features_after_imputer(imputer_features,pass_features).split(","))],
                                                remainder = "drop")
    col_transformer.set_output(transform='pandas')

    pipeline = Pipeline([("imputer",imputer),
                        ("col_transformer",col_transformer),
                        ("clf",LGBMClassifier(learning_rate = grid["learning_rate"],
                                            n_estimators =  grid["n_estimators"],
                                            max_depth = grid["max_depth"],
                                            num_leaves = grid["num_leaves"],
                                            min_child_samples = grid["min_child_samples"],
                                            reg_alpha = grid["reg_alpha"],
                                            reg_lambda = grid["reg_lambda"],
                                            verbose=-1,
                                            class_weight= "balanced",
                                            random_state=seed))])
    return pipeline