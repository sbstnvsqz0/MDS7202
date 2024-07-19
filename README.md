En esta carpeta se subirá todo lo relacionado con laboratorios y proyectos del curso Laboratorio de Programación Científica para Ciencia de Datos.

Para utilizar optimization.py se hace lo siguiente:

1. Se abre terminal con environment configurado
2. Se ejecuta comando como sigue:
python optimization.py --name (rfc,xgb,lgbm) --data (.csv con data X) --label (.csv con data y) --trials (número de trials que se quieren hacer)
Un ejemplo sería:
python optimization.py --name xgb --data X_t0 --label y_t0 --trials 100