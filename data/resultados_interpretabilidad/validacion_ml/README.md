# Validación de los modelos y de SHAP individual

Se verificaron los nueve modelos originales de la semilla 42, sus parámetros seleccionados,
sus predicciones y las tres votaciones. No se entrenaron modelos. Las 7 200 etiquetas del
test completo coinciden; el máximo error de probabilidades es inferior a 1e-6.

La reconstrucción de SHAP considera los 598, 577 y 599 textos filtrados, por cada uno de
los tres modelos. Todas las salidas se reconstruyen con error inferior a 1e-5. El mayor
error es aproximadamente 2.73e-6, en el margen de XGBoost. Los CSV conservan también la
tolerancia inicial de 1e-3 del cuaderno preparado y los conteos de errores de 1e-6 y 1e-5.

`global_recalculado.csv` y `comparacion_global_guardado.csv` contrastan los valores
publicados con los obtenidos a partir de estos modelos. No se sobrescribieron las
explicaciones originales. Las referencias de RF, los errores por texto y las huellas
de los datos y modelos se conservan en esta carpeta.

La reproducción de la aditividad utiliza `notebooks/07b_Verificar_SHAP_ML.ipynb`. 07a reproduce las predicciones y calcula
SHAP de la votación. La comprobación de aditividad no convierte métodos y referencias
diferentes en directamente equivalentes ni demuestra la causa de un fallo previo.
