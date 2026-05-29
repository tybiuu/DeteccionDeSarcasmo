# Fase 1 - Carga y Preprocesamiento del Dataset IroSvA
Se aplica la secuencia de preprocesamiento común a todos los modelos y se extraen
las 9 variables lingüísticas como features numéricos. Se utiliza la partición oficial
train/test del dataset (Ortega-Bueno et al., 2019).
Se generan archivos combinados (3 variantes) y por variante individual.

## 1. Importación de librerías

```python
import pandas as pd
import re
import emoji
import unicodedata
```

## 2. Carga y unión del conjunto de entrenamiento
Se combinan las tres variantes dialectales (México, España, Cuba) y se eliminan
los 3 duplicados identificados en la fase de exploración.

```python
df_mx_train = pd.read_csv('../data/irosva.mx.training.csv')
df_es_train = pd.read_csv('../data/irosva.es.training.csv')
df_cu_train = pd.read_csv('../data/irosva.cu.training.csv')

df_mx_train['VARIANTE'] = 'mx'
df_es_train['VARIANTE'] = 'es'
df_cu_train['VARIANTE'] = 'cu'

df_train = pd.concat([df_mx_train, df_es_train, df_cu_train], ignore_index=True)
antes = len(df_train)
df_train = df_train.drop_duplicates(subset='MESSAGE').reset_index(drop=True)
eliminados = antes - len(df_train)
print(f'Train: {antes} → {len(df_train)} registros ({eliminados} duplicados eliminados)')
print(f'Registros por variante:')
print(df_train['VARIANTE'].value_counts().to_dict())
```

## 3. Carga y combinación del conjunto de test
Los archivos de test del dataset IroSvA separan el texto de las etiquetas reales
en dos archivos distintos. Se combinan por ID antes de unir las variantes.

```python
def cargar_test(variante, nombre_variante):
    df_texto = pd.read_csv(f'../data/irosva.{variante} - irosva.{variante}.test.csv')
    df_truth = pd.read_csv(f'../data/irosva.{variante}.test.truth.csv')
    df_texto = df_texto.drop(columns=['IS_IRONIC'])
    df = df_texto.merge(df_truth[['ID', 'IS_IRONIC']], on='ID')
    df['VARIANTE'] = nombre_variante
    return df

df_test = pd.concat([
    cargar_test('mx', 'mx'),
    cargar_test('es', 'es'),
    cargar_test('cu', 'cu')
], ignore_index=True)

# Verificar duplicados en test
dups_test = df_test.duplicated(subset='MESSAGE').sum()
print(f'Test: {len(df_test)} registros | {dups_test} duplicados en MESSAGE')
print(f'Distribución IS_IRONIC: {df_test["IS_IRONIC"].value_counts().to_dict()}')
print(f'Registros por variante:')
print(df_test['VARIANTE'].value_counts().to_dict())
```

## 4. Extracción de variables lingüísticas

Se extraen **9 variables lingüísticas** del texto original **antes del preprocesamiento**
para capturar cada señal en su forma natural, antes de que la limpieza la transforme
o elimine.

| Feature | Extraer antes porque... |
|---|---|
| n_exc, n_int | Se conservan en preprocesamiento ✓ |
| n_may | No se aplica lowercase ✓ |
| n_emo | Paso 3 convierte emojis a texto |
| n_ris | No se eliminan en preprocesamiento ✓ |
| n_neg | No se eliminan palabras ✓ |
| n_elo | Paso 4 normaliza a máx. 2 repeticiones |
| n_com | Paso 5 elimina comillas |
| n_pun | Paso 4 normaliza '...' a '..' |

(González-Ibáñez et al., 2011; Joshi et al., 2015, 2017;
Šandor & Bagić Babac, 2023; RAE, 2011)

```python
# ── UTILIDAD: normalización de tildes ────────────────────────────────────────
def quitar_tildes(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

# ── LISTA DE NEGACIONES EN ESPAÑOL (RAE, 2011) ───────────────────────────────
# Adverbios: no, nunca, jamás, tampoco
# Pronombres/determinantes: nada, nadie, ninguno/a/os/as, ningún
# Conjunciones: ni, sino
# Preposición: sin
# Almacenadas sin tildes para comparar sobre texto normalizado.
NEGACIONES = {
    'no', 'nunca', 'jamas', 'tampoco',
    'nada', 'nadie',
    'ningun', 'ninguno', 'ninguna', 'ningunos', 'ningunas',
    'ni', 'sino',
    'sin'
}
```

```python
def extraer_features_linguisticos(texto):
    """
    Extrae 9 variables lingüísticas como conteos sobre el texto original
    sin preprocesar. Debe ejecutarse ANTES de preprocesar().
    """
    texto_str  = str(texto)
    texto_norm = quitar_tildes(texto_str.lower())
    palabras   = re.findall(r'\b\w+\b', texto_norm)

    # 1. Exclamaciones: ! y ¡
    n_exc = texto_str.count('!') + texto_str.count('¡')
    # 2. Interrogaciones: ? y ¿
    n_int = texto_str.count('?') + texto_str.count('¿')
    # 3. Mayúsculas enfáticas: palabras de 3+ chars en mayúsculas
    n_may = len(re.findall(r'\b[A-ZÁÉÍÓÚÜÑ]{3,}\b', texto_str))
    # 4. Emojis: antes de conversión a texto (paso 3 preprocesamiento)
    n_emo = sum(1 for c in texto_str if c in emoji.EMOJI_DATA)
    # 5. Risas
    n_ris = len(re.findall(
        r'\b(ja+ja+|je+je+|ji+ji+|ha+ha+|xs+|xd+|lol+|jsjs)\b',
        texto_norm
    ))
    # 6. Negaciones: RAE (2011), comparación sin tildes
    n_neg = sum(1 for p in palabras if p in NEGACIONES)
    # 7. Elongación: 3+ repeticiones (antes de normalización paso 4)
    n_elo = len(re.findall(r'(.)\1{2,}', texto_norm))
    # 8. Comillas: antes de eliminación de chars especiales (paso 5)
    n_com = len(re.findall(r'["\'\u2018\u2019\u201c\u201d«»]', texto_str))
    # 9. Puntos suspensivos: antes de normalización (paso 4)
    n_pun = texto_str.count('...') + texto_str.count('\u2026')

    return [n_exc, n_int, n_may, n_emo, n_ris, n_neg, n_elo, n_com, n_pun]

FEATURE_COLS = ['n_exc','n_int','n_may','n_emo','n_ris',
                'n_neg','n_elo','n_com','n_pun']

# Extraer features ANTES del preprocesamiento en ambos conjuntos
for df in [df_train, df_test]:
    feats = df['MESSAGE'].apply(extraer_features_linguisticos).tolist()
    for i, col in enumerate(FEATURE_COLS):
        df[col] = [f[i] for f in feats]

print(f'Features extraídos: {FEATURE_COLS}')
print('\n=== Medias features — Train ===')
print(df_train[FEATURE_COLS].mean().round(3))
print('\n=== Medias features — Test ===')
print(df_test[FEATURE_COLS].mean().round(3))
```

## 5. Función de preprocesamiento

Se conservan mayúsculas enfáticas y puntuación expresiva como señales pragmáticas
de ironía (Ortega-Bueno et al., 2022). Los emojis se convierten a descripciones
textuales en español para preservar su carga semántica (Qin et al., 2025).
Las menciones y URLs se reemplazan por tokens neutros.

**Consistencia con features:** los pasos 3, 4 y 5 modifican señales ya capturadas:
paso 3 → emojis (n_emo extraído) | paso 4 → elongación y puntos suspensivos
(n_elo, n_pun extraídos) | paso 5 → comillas (n_com extraído).

```python
def preprocesar(texto):
    # 1. Reemplazar URLs
    texto = re.sub(r'http\S+|www\S+', '[URL]', texto)
    # 2. Reemplazar menciones @usuario
    texto = re.sub(r'@\w+', '[USER]', texto)
    # 3. Convertir emojis a descripción textual en español (n_emo ya extraído)
    texto = emoji.demojize(texto, language='es')
    # 4. Normalizar caracteres repetidos: máx. 2 (n_elo, n_pun ya extraídos)
    texto = re.sub(r'(.)\1{2,}', r'\1\1', texto)
    # 5. Eliminar toda la puntuación (features ya extraídos antes)
    texto = re.sub(r'[^\w\s\[\]áéíóúüñÁÉÍÓÚÜÑ]', ' ', texto)
    # 6. Convertir a minúsculas (compatibilidad con TF-IDF y GloVe)
    texto = texto.lower()
    # 7. Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto
```

## 6. Aplicación del preprocesamiento y verificación

```python
df_train['MESSAGE_CLEAN'] = df_train['MESSAGE'].apply(preprocesar)
df_test['MESSAGE_CLEAN']  = df_test['MESSAGE'].apply(preprocesar)
print(f'Preprocesamiento aplicado — Train: {df_train.shape} | Test: {df_test.shape}')

# Verificación con un ejemplo por variante
for variante in ['mx', 'es', 'cu']:
    ejemplo = df_train[df_train['VARIANTE'] == variante].iloc[0]
    print(f'\n[{variante.upper()}]')
    print(f'ORIGINAL: {ejemplo["MESSAGE"]}')
    print(f'LIMPIO:   {ejemplo["MESSAGE_CLEAN"]}')
```

## 7. Exportación de datos preprocesados

Se exportan dos tipos de archivos:
- **Combinados**: las 3 variantes unidas, para entrenamiento general.
- **Por variante**: cada variante por separado, para análisis dialectal.

Los archivos por variante se derivan del dataset combinado ya procesado
(duplicados eliminados, features extraídos, preprocesamiento aplicado),
por lo que son completamente consistentes con el combinado.

```python
# ── ARCHIVOS COMBINADOS (3 variantes) ────────────────────────────────────────
df_train.to_csv('../data/train_clean.csv', index=False)
df_test.to_csv('../data/test_clean.csv', index=False)
print('Combinados:')
print(f'  train_clean.csv  → {len(df_train):,} registros')
print(f'  test_clean.csv   → {len(df_test):,} registros')

# ── ARCHIVOS POR VARIANTE ─────────────────────────────────────────────────────
# Se filtran del dataset combinado ya procesado: no se repite ningún paso.
# La distribución de clases y features es consistente con el combinado.
VARIANTES = {
    'mx': 'México',
    'es': 'España',
    'cu': 'Cuba'
}

print('\nPor variante:')
for cod, nombre in VARIANTES.items():
    train_v = df_train[df_train['VARIANTE'] == cod].reset_index(drop=True)
    test_v  = df_test[df_test['VARIANTE']  == cod].reset_index(drop=True)

    train_v.to_csv(f'../data/train_clean_{cod}.csv', index=False)
    test_v.to_csv(f'../data/test_clean_{cod}.csv',  index=False)

    ironico_train = train_v['IS_IRONIC'].sum()
    ironico_test  = test_v['IS_IRONIC'].sum()
    print(f'  {nombre:8} train_{cod}: {len(train_v):,} registros '
          f'(irónico: {ironico_train}, no irónico: {len(train_v)-ironico_train})')
    print(f'  {"":8} test_{cod}:  {len(test_v):,} registros '
          f'(irónico: {ironico_test}, no irónico: {len(test_v)-ironico_test})')

print(f'\nColumnas exportadas: {df_train.columns.tolist()}')
```

## 8. Resumen de archivos generados
```
data/
├── train_clean.csv        ← 3 variantes combinadas (train)
├── test_clean.csv         ← 3 variantes combinadas (test)
├── train_clean_mx.csv     ← México (train)
├── test_clean_mx.csv      ← México (test)
├── train_clean_es.csv     ← España (train)
├── test_clean_es.csv      ← España (test)
├── train_clean_cu.csv     ← Cuba (train)
└── test_clean_cu.csv      ← Cuba (test)
```