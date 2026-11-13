# Figuras SHAP del artículo

Ejecutar la configuración inicial y las secciones 5 y 6 de `notebooks/07_Interpretabilidad.ipynb`.

La Figura 6.2 utiliza los ganadores del F1-Macro medio del test filtrado por familia y variante:
RoBERTuito en las tres variantes, RL en México y votación de ML en España y Cuba. El notebook 07a
verifica las predicciones y calcula las contribuciones de la decisión completa de la votación.
Sus resultados se guardan en `data/resultados_interpretabilidad/votacion_ml`. Si ya están disponibles,
07a los valida y reutiliza con `RECALCULAR=False`; después, 07 lee esos CSV y genera la figura.
RoBERTuito comparte escala entre sus tres paneles y la votación entre España y Cuba. RL conserva
su escala de log-odds. Las notas externas distinguen las unidades y la referencia del cálculo.

La Figura 6.3 utiliza tres casos de RoBERTuito y hasta diez contribuciones de mayor magnitud
absoluta por caso, con desempate por posición. La presentación conserva el orden del texto
y no agrupa las palabras repetidas. México muestra sus seis contribuciones, España diez de
26 y Cuba diez de 30. Las palabras omitidas pueden tener una contribución no nula.

Los tres paneles se presentan en una fila. Comparten el rango de −0.27 a +0.27 y marcas en
−0.20, 0 y +0.20. Un control impide guardar valores fuera del rango. La salida mide 6.5 por
3.2 pulgadas. Las palabras usan 8.5 puntos y los valores 8 puntos, con filas de 10.8 puntos.
Las anotaciones numéricas que exceden el eje se colocan dentro de la barra.

Las transcripciones completas acompañan la imagen fuera de ella, identificadas con las letras
a, b y c, y se exportan en `shap_local_textos_articulo.md`. El HTML conserva las 62 contribuciones
en tablas desplegables. El CSV mantiene los mismos valores, etiquetas y probabilidades;
`mostrada_en_barras` identifica las 26 contribuciones de la figura compacta.

La versión completa anterior queda en `shap_local_completo.png`, `.svg` y `.html`. Es una copia
de respaldo: regenerar la versión compacta no la sobrescribe. Las versiones anteriores también
permanecen en los respaldos de la actualización.

PNG permite insertar la figura y SVG conserva texto y trazos vectoriales. Las notas externas
figuran en `Notas_figuras_articulo.md`. Las barras no incluyen el valor de referencia y no deben
sumarse por sí solas para reconstruir la probabilidad. No se entrenan modelos ni se recalcula
SHAP para este cambio. Los ejemplos mantienen su alcance ilustrativo.
