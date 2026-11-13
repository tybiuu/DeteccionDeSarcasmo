# SHAP de la votación de ML

Este análisis explica la decisión binaria de mayoría en España y Cuba de la semilla 42.
No explica probabilidades ni promedia SHAP de componentes. El cuaderno reproducible es
`notebooks/07a_SHAP_votacion_ML.ipynb`. El notebook 07 lee estos resultados para la figura.

Se explican los 577 textos de España y los 599 de Cuba del test filtrado común, con 128
permutaciones bidireccionales por texto. Las palabras se eliminan de forma simultánea
de las representaciones de sus componentes; las formas preprocesadas permanecen fijas.
Los nueve rasgos se consideran jugadores adicionales y se contrastan con cero. La
referencia es texto vacío y rasgos en cero: su decisión es 1 en España y 0 en Cuba.
Cada explicación reconstruye la decisión original con error inferior a 1e-8.

Los CSV contienen valores por caso, resúmenes léxicos y lingüísticos y comprobaciones
de aditividad. `configuracion_shap_votacion.json` registra versiones, huellas, referencia,
semillas, filtro y el control exploratorio de 128 frente a 256 permutaciones. Los resúmenes
léxicos promedian apariciones; `n_textos` cuenta los IDs distintos.

La comparación de magnitudes con otros métodos o referencias no es directa. El análisis
es exploratorio y no prueba mecanismos causales ni frecuencias de comprensión pragmática.
Las etiquetas y probabilidades de los nueve modelos originales se comprobaron por separado.

La ejecución verificada utilizó Python 3.11, numpy 1.26.4, pandas 3.0.3, scikit-learn 1.8.0,
XGBoost 3.2.0, SHAP 0.49.1, spaCy 3.7.5, es_core_news_sm 3.7.0 y NLTK 3.9.4. Utilizar
ese entorno para reproducir exactamente el análisis; el cuaderno no instala dependencias.
Con `RECALCULAR=False` valida y reutiliza las salidas completas ya existentes.
