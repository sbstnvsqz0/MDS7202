import pickle
import shap 
from sklearn.pipeline import Pipeline
from utils import eval_pipe
from lightgbm import LGBMClassifier
from utils import optimize_lgbm, pipe_lgbm

class Clf():
    def __init__(self,name,model=None,t=None):
        self.model = model
        self.name = name
        self.t = t

    def load(self,path,t):
        """Carga el modelo mediante un path y un tiempo"""
        with open(path, 'rb') as file:
            self.model = pickle.load(file)
        self.t = t
    
    def save(self):
        """Guarda el modelo en un path"""
        assert self.model is not None and self.t is not None, "No se ha cargado un modelo"
        with open(f'{self.name}_{self.t}.pkl','wb') as file:
            pickle.dump(self.model,file)        
    
    def shap(self,X):
        """Calcula las explicaciones de las predicciones"""
        assert self.model is not None and self.t is not None, "No se ha cargado un modelo"
        transform_data = Pipeline(self.model.steps[0:2])    #Imputer y col_transformer se dejan igual
        explainer = shap.Explainer(self.model.steps[-1][1]) #Se toma el modelo
        shap_values = explainer(transform_data.transform(X))
        shap.plots.waterfall(shap_values[0])    #Se muestran los atributos que más influyeron en la clasificación de la muestra 0.
        shap.summary_plot(shap_values, transform_data.transform(X)) #Resumen de cuanto puede influir un atributo
    
    def eval(self,X,y):
        """Evalua el modelo sobre datos X y labels y"""
        eval_pipe(self.model,X,y)
    
    def retrain(self,X,y,t):
        """Entrena el modelo con datos 'X' y labels 'y' a partir del modelo anterior"""
        transform_data = Pipeline(self.model.steps[0:2])
        X = transform_data.transform(X)
        new_model = LGBMClassifier(**self.model.steps[-1][1].get_params())
        new_model.fit(X,y,init_model=self.model[-1])

        self.model.steps.pop(-1)
        self.model.steps.append(("clf",new_model))
        self.t = t

    def optim(self,X_train,X_val,y_train,y_val,trials):
        """
        Optimiza el modelo guardado
        """
        assert self.model is not None and self.t is not None, "No se ha cargado un modelo"
        best_params = optimize_lgbm(self.model[-1],X_train,X_val,y_train,y_val,trials)
        self.model = pipe_lgbm(best_params)

    def time(self):
        """Devuelve el ultimo tiempo en el que se entrenó el modelo"""
        return self.t