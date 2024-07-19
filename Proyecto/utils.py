from plotly.subplots import make_subplots
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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