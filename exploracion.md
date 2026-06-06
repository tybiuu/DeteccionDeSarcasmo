# Fase 0 - Exploración del Dataset IroSvA
Análisis estadístico descriptivo de las tres variantes dialectales del corpus IroSvA
(Ortega-Bueno et al., 2019): distribución de clases, presencia de variables lingüísticas
relevantes para detección de ironía y análisis del vocabulario.
Los resultados se presentan por variante dialectal y para el corpus completo.

## 1. Importación de librerías

```python
import pandas as pd
import re
import emoji
import unicodedata
```

## 2. Carga de las tres variantes dialectales

```python
df_mx = pd.read_csv('../data/irosva.mx.training.csv')
df_es = pd.read_csv('../data/irosva.es.training.csv')
df_cu = pd.read_csv('../data/irosva.cu.training.csv')

# Dataset combinado
df_all = pd.concat([df_mx, df_es, df_cu], ignore_index=True)

# Estructura unificada para iterar sobre variantes + total
DATASETS = [
    ('México',              df_mx),
    ('España',              df_es),
    ('Cuba',                df_cu),
    ('TOTAL (3 variantes)', df_all)
]

print(f'Columnas: {df_mx.columns.tolist()}')
print(f'Tipos:    {df_mx.dtypes.to_dict()}')
print(df_mx.head(3))
```

## 3. Validación de calidad: nulos y duplicados

```python
for nombre, df in DATASETS:
    nulos = df.isnull().sum().sum()
    dups  = df.duplicated(subset='MESSAGE').sum()
    print(f'{nombre:25} {nulos} nulos | {dups} duplicados en MESSAGE')
```

## 4. Distribución de clases
El dataset presenta un desbalance 2:1 (no irónico:irónico) consistente
en las tres variantes (Ortega-Bueno et al., 2019).

```python
print(f'{"Variante":25} {"Total":>7} {"Irónico":>10} {"No irónico":>12}')
print('-' * 58)
for nombre, df in DATASETS:
    total      = len(df)
    ironico    = df['IS_IRONIC'].sum()
    no_ironico = total - ironico
    print(f'{nombre:25} {total:>7,} '
          f'{ironico:>5} ({ironico/total*100:.1f}%) '
          f'{no_ironico:>5} ({no_ironico/total*100:.1f}%)')
```

## 5. Ejemplos de tweets irónicos y no irónicos

```python
for nombre, df in DATASETS[:-1]:  # solo las 3 variantes, no el total
    print(f'\n=== {nombre.upper()} — TWEETS IRÓNICOS ===')
    for tweet in df[df['IS_IRONIC']==1]['MESSAGE'].head(3).values:
        print(' -', tweet)
    print(f'\n=== {nombre.upper()} — TWEETS NO IRÓNICOS ===')
    for tweet in df[df['IS_IRONIC']==0]['MESSAGE'].head(3).values:
        print(' -', tweet)
```

## 6. Extracción y análisis de variables lingüísticas por clase

Se analizan 9 variables lingüísticas asociadas a la detección de sarcasmo:
exclamaciones, interrogaciones, mayúsculas enfáticas, emojis, risas, negaciones,
elongación de caracteres, comillas y puntos suspensivos
(González-Ibáñez et al., 2011; Joshi et al., 2015, 2017; Šandor & Bagić Babac, 2023).

Todas se extraen del texto original **antes del preprocesamiento** para capturar
emojis, elongación y signos de puntuación en su forma natural.

```python
# ── UTILIDAD: normalización de tildes ────────────────────────────────────────
# Convierte caracteres acentuados a su forma base (jamás→jamas, ningún→ningun)
# para capturar variantes de escritura informal propias de redes sociales.
def quitar_tildes(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

# ── LISTA DE NEGACIONES EN ESPAÑOL ───────────────────────────────────────────
# Fuente: Real Academia Española y Asociación de Academias de la Lengua
# Española (2011). Nueva gramática básica de la lengua española [en línea].
# https://www.rae.es/gramatica-basica/la-modalidad-la-negacion/la-negacion/concepto
#
# Clasificación oficial:
#   Adverbios:               no, nunca, jamás, tampoco, nada (uso adverbial)
#   Pronombres/determinantes: nadie, ninguno/a/os/as, ningún, nada
#   Conjunciones:            ni, sino
#   Preposición:             sin
#
# Se almacenan sin tildes porque la comparación se realiza sobre texto
# normalizado con quitar_tildes().
NEGACIONES = {
    # Adverbios de negación
    'no', 'nunca', 'jamas', 'tampoco',
    # Pronombres y determinantes negativos
    'nada', 'nadie',
    'ningun', 'ninguno', 'ninguna', 'ningunos', 'ningunas',
    # Conjunciones negativas
    'ni', 'sino',
    # Preposición con valor negativo
    'sin'
}

# ── NOMBRES DE FEATURES ──────────────────────────────────────────────────────
FEATURES = ['n_exc','n_int','n_may','n_emo','n_ris','n_neg','n_elo','n_com','n_pun']
NOMBRES  = [
    'Exclamaciones', 'Interrogaciones', 'Mayúsculas enfáticas',
    'Emojis', 'Risas', 'Negaciones',
    'Elongación', 'Comillas', 'Puntos suspensivos'
]
```

```python
def extraer_features(texto):
    """
    Extrae 9 variables lingüísticas como conteos sobre el texto original
    sin preprocesar. Retorna int para uso directo en modelos ML.
    """
    texto_str  = str(texto)
    texto_norm = quitar_tildes(texto_str.lower())
    palabras   = re.findall(r'\b\w+\b', texto_norm)

    # 1. Exclamaciones: signos ! y ¡
    # (González-Ibáñez et al., 2011; Joshi et al., 2015)
    n_exc = texto_str.count('!') + texto_str.count('¡')

    # 2. Interrogaciones: signos ? y ¿
    # (González-Ibáñez et al., 2011; Joshi et al., 2015)
    n_int = texto_str.count('?') + texto_str.count('¿')

    # 3. Mayúsculas enfáticas: palabras de 3+ chars en mayúsculas
    # Umbral de 3 chars reduce ruido de siglas de 2 letras (RT, PP)
    # (Joshi et al., 2015; Šandor & Bagić Babac, 2023)
    n_may = len(re.findall(r'\b[A-ZÁÉÍÓÚÜÑ]{3,}\b', texto_str))

    # 4. Emojis: antes de conversión a texto en preprocesamiento
    # (González-Ibáñez et al., 2011; Joshi et al., 2015)
    n_emo = sum(1 for c in texto_str if c in emoji.EMOJI_DATA)

    # 5. Risas: expresiones en español e inglés
    # (González-Ibáñez et al., 2011)
    n_ris = len(re.findall(
        r'\b(ja+ja+|je+je+|ji+ji+|ha+ha+|xs+|xd+|lol+|jsjs)\b',
        texto_norm
    ))

    # 6. Negaciones: clasificación RAE (2011), comparación sobre texto
    # normalizado sin tildes para cubrir escritura informal
    # (González-Ibáñez et al., 2011; RAE, 2011)
    n_neg = sum(1 for p in palabras if p in NEGACIONES)

    # 7. Elongación: mismo carácter repetido 3+ veces consecutivas
    # Extraído antes de normalización de caracteres repetidos
    # (Šandor & Bagić Babac, 2023; Joshi et al., 2017)
    n_elo = len(re.findall(r'(.)\1{2,}', texto_norm))

    # 8. Comillas: distanciamiento irónico (dobles, simples, españolas «»)
    # Extraído antes de eliminación de caracteres especiales
    # (Šandor & Bagić Babac, 2023)
    n_com = len(re.findall(r'["\'\u2018\u2019\u201c\u201d«»]', texto_str))

    # 9. Puntos suspensivos: tono suspensivo irónico ('...', '…')
    # Extraído antes de eliminación de puntuación
    # (Šandor & Bagić Babac, 2023)
    n_pun = texto_str.count('...') + texto_str.count('\u2026')

    return {
        'n_exc': n_exc, 'n_int': n_int, 'n_may': n_may,
        'n_emo': n_emo, 'n_ris': n_ris, 'n_neg': n_neg,
        'n_elo': n_elo, 'n_com': n_com, 'n_pun': n_pun
    }
```

```python
def analizar_features(nombre, df):
    """Extrae features y muestra presencia y media por clase para un dataset."""
    feats = df['MESSAGE'].apply(extraer_features).apply(pd.Series)
    df_f  = pd.concat([df[['IS_IRONIC']], feats], axis=1)

    print(f'\n{"="*58}')
    print(f'  {nombre}')
    print(f'{"="*58}')

    # Presencia (% textos con al menos 1 ocurrencia)
    print(f'\n  Presencia (% de textos con al menos 1 ocurrencia):')
    print(f'  {"Variable":23} {"No irónico":>12} {"Irónico":>10}')
    print(f'  {"-"*48}')
    for feat, nom in zip(FEATURES, NOMBRES):
        p = df_f.groupby('IS_IRONIC')[feat].apply(lambda x: (x>0).mean()*100)
        print(f'  {nom:23} {p[0]:>11.1f}% {p[1]:>9.1f}%')

    # Media de ocurrencias por texto
    print(f'\n  Media de ocurrencias por texto:')
    print(f'  {"Variable":23} {"No irónico":>12} {"Irónico":>10}')
    print(f'  {"-"*48}')
    for feat, nom in zip(FEATURES, NOMBRES):
        m = df_f.groupby('IS_IRONIC')[feat].mean()
        print(f'  {nom:23} {m[0]:>12.3f} {m[1]:>10.3f}')

    return df_f

# Ejecutar para cada variante y el total
resultados = {}
for nombre, df in DATASETS:
    resultados[nombre] = analizar_features(nombre, df)
```

## 7. Análisis de vocabulario
Se analiza el tamaño del vocabulario por variante dialectal y en el corpus completo
para justificar el límite de 10 000 términos en la vectorización TF-IDF
(Pandey & Singh, 2023; Šandor & Bagić Babac, 2023).

```python
def tokenizar_simple(texto):
    """Tokenización básica: minúsculas, solo palabras alfanuméricas."""
    return re.findall(r'\b\w+\b', str(texto).lower())

print('Tamaño del vocabulario por variante (texto crudo, sin preprocesamiento):\n')
print(f'{"Variante":25} {"Tokens totales":>16} {"Vocabulario único":>18}')
print('-' * 62)

vocab_total = set()
for nombre, df in DATASETS[:-1]:  # solo las 3 variantes
    tokens = [t for msg in df['MESSAGE'] for t in tokenizar_simple(msg)]
    vocab  = set(tokens)
    vocab_total.update(vocab)
    print(f'{nombre:25} {len(tokens):>16,} {len(vocab):>18,}')

print('-' * 62)
# Total: usa set para no contar palabras compartidas entre variantes
tokens_all = [t for msg in df_all['MESSAGE'] for t in tokenizar_simple(msg)]
print(f'{"TOTAL (3 variantes)":25} {len(tokens_all):>16,} {len(vocab_total):>18,}')
print(f'\nEl límite de 10 000 términos en TF-IDF retiene el '
      f'{min(10000/len(vocab_total)*100, 100):.1f}% del vocabulario total.')
```

```python
# Distribución de longitud de textos (en tokens)
print('Longitud de textos (tokens) por variante:\n')
print(f'{"Variante":25} {"Min":>5} {"Media":>7} {"Mediana":>8} {"Max":>6} {"P90":>7}')
print('-' * 62)

for nombre, df in DATASETS:
    lon = df['MESSAGE'].apply(lambda x: len(tokenizar_simple(x)))
    print(f'{nombre:25} {lon.min():>5} {lon.mean():>7.1f} '
          f'{lon.median():>8.1f} {lon.max():>6} {lon.quantile(0.9):>7.1f}')
```