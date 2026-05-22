# Fase 0 - Exploración del Dataset IroSvA
Análisis estadístico descriptivo de las tres variantes dialectales del corpus IroSvA (Ortega-Bueno et al., 2019): distribución de clases y presencia de variables lingüísticas relevantes para detección de ironía.

## 1. Importación de librerías

```python
import pandas as pd
import re
import emoji
```

## 2. Carga de las tres variantes dialectales

```python
df_mx = pd.read_csv('../data/irosva.mx.training.csv')
df_es = pd.read_csv('../data/irosva.es.training.csv')
df_cu = pd.read_csv('../data/irosva.cu.training.csv')

print(f"Columnas: {df_mx.columns.tolist()}")
print(f"Tipos:    {df_mx.dtypes.to_dict()}")
print(df_mx.head(3))
```

## 3. Validación de calidad: nulos y duplicados

```python
for nombre, df in [('México', df_mx), ('España', df_es), ('Cuba', df_cu)]:
    nulos = df.isnull().sum().sum()
    dups  = df.duplicated(subset='MESSAGE').sum()
    print(f"{nombre}: {nulos} nulos | {dups} duplicados en MESSAGE")
```

## 4. Distribución de clases por variante
El dataset presenta un desbalance 2:1 (no irónico:irónico) consistente en las tres variantes (Ortega-Bueno et al., 2019).

```python
for nombre, df in [('México', df_mx), ('España', df_es), ('Cuba', df_cu)]:
    total      = len(df)
    ironico    = df['IS_IRONIC'].sum()
    no_ironico = total - ironico
    print(f"{nombre}: Total={total} | Irónico={ironico} ({ironico/total*100:.1f}%) | No irónico={no_ironico} ({no_ironico/total*100:.1f}%)")
```

## 5. Ejemplos de tweets irónicos y no irónicos

```python
print("=== TWEETS IRÓNICOS ===")
for tweet in df_mx[df_mx['IS_IRONIC']==1]['MESSAGE'].head(5).values:
    print("-", tweet)

print("\n=== TWEETS NO IRÓNICOS ===")
for tweet in df_mx[df_mx['IS_IRONIC']==0]['MESSAGE'].head(5).values:
    print("-", tweet)
```

## 6. Análisis de variables lingüísticas por clase
Se analizan 7 variables lingüísticas asociadas a la detección de sarcasmo en la literatura: exclamaciones, interrogaciones, mayúsculas enfáticas, emojis, risas, negaciones e incongruencia sentimental (González-Ibáñez et al., 2011; Joshi et al., 2015; Ortega-Bueno et al., 2022). Se extraen del texto original antes del preprocesamiento para capturar emojis y patrones en su forma natural.

```python
# Lexicons simples para incongruencia sentimental
NEGACIONES = {'no','nunca','jamás','ni','tampoco','nada','nadie','ningún','sin'}

def analizar_features(texto):
    texto_str  = str(texto)
    texto_lower = texto_str.lower()
    palabras   = re.findall(r'\b\w+\b', texto_lower)
    return {
        'exclamaciones':   texto_str.count('!') > 0,
        'interrogaciones': texto_str.count('?') > 0,
        'mayusculas':      bool(re.search(r'\b[A-ZÁÉÍÓÚÜÑ]{3,}\b', texto_str)),
        'emojis':          any(c in emoji.EMOJI_DATA for c in texto_str),
        'risas':           bool(re.search(r'\b(ja+ja+|je+je+|ji+ji+|ha+ha+|xs+|xd+|lol+|jsjs)\b', texto_lower)),
        'negaciones':      sum(1 for p in palabras if p in NEGACIONES) > 0
    }

df_all = pd.concat([df_mx, df_es, df_cu], ignore_index=True)

features_df = df_all['MESSAGE'].apply(analizar_features).apply(pd.Series)
df_all = pd.concat([df_all, features_df], axis=1)

print("Presencia de variables lingüísticas por clase (%):\n")
for feature in ['exclamaciones','interrogaciones','mayusculas','emojis','risas','negaciones']:
    por_clase = df_all.groupby('IS_IRONIC')[feature].mean() * 100
    print(f"{feature:20} No irónico: {por_clase[0]:.1f}% | Irónico: {por_clase[1]:.1f}%")
```
