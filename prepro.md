# Fase 1 — Preprocesamiento del Dataset IroSvA
Se aplica la secuencia de preprocesamiento común a todos los modelos y se extraen
las 9 variables lingüísticas como features numéricos. Se utiliza la partición oficial
train/test del dataset (Ortega-Bueno et al., 2019).

Se generan archivos por variante dialectal (mx, es, cu) para tres variantes
de texto: normal, stemming y lematización.

## 1. Importación de librerías

```python
import pandas as pd
import re
import emoji
import unicodedata
import nltk
from nltk.stem import SnowballStemmer
import spacy
from tqdm import tqdm

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
```

## 2. Carga del conjunto de entrenamiento por variante
Se cargan las tres variantes dialectales por separado. La deduplicación
se realiza dentro de cada variante para preservar la independencia entre corpus.

```python
df_mx_train = pd.read_csv('../data/irosva.mx.training.csv')
df_es_train = pd.read_csv('../data/irosva.es.training.csv')
df_cu_train = pd.read_csv('../data/irosva.cu.training.csv')

df_mx_train['VARIANTE'] = 'mx'
df_es_train['VARIANTE'] = 'es'
df_cu_train['VARIANTE'] = 'cu'

# Deduplicación por variante
for nombre, df in [('mx', df_mx_train), ('es', df_es_train), ('cu', df_cu_train)]:
    antes = len(df)
    dups = df.duplicated(subset='MESSAGE').sum()
    print(f'  {nombre}: {antes} registros | {dups} duplicados en MESSAGE')

df_mx_train = df_mx_train.drop_duplicates(subset='MESSAGE').reset_index(drop=True)
df_es_train = df_es_train.drop_duplicates(subset='MESSAGE').reset_index(drop=True)
df_cu_train = df_cu_train.drop_duplicates(subset='MESSAGE').reset_index(drop=True)

# Combinar para procesamiento eficiente (stemming/lematización en batch)
df_train = pd.concat([df_mx_train, df_es_train, df_cu_train], ignore_index=True)
print(f'\nTotal train tras deduplicación por variante: {len(df_train)}')
print(f'Registros por variante: {df_train["VARIANTE"].value_counts().to_dict()}')
```

## 3. Carga del conjunto de test por variante
Los archivos de test del dataset IroSvA separan el texto de las etiquetas
en dos archivos distintos. Se combinan por ID.

```python
def cargar_test(variante):
    df_texto = pd.read_csv(f'../data/irosva.{variante} - irosva.{variante}.test.csv')
    df_truth = pd.read_csv(f'../data/irosva.{variante}.test.truth.csv')
    df_texto = df_texto.drop(columns=['IS_IRONIC'])
    df = df_texto.merge(df_truth[['ID', 'IS_IRONIC']], on='ID')
    df['VARIANTE'] = variante
    return df

df_mx_test = cargar_test('mx')
df_es_test = cargar_test('es')
df_cu_test = cargar_test('cu')

df_test = pd.concat([df_mx_test, df_es_test, df_cu_test], ignore_index=True)

print(f'Test: {len(df_test)} registros')
for variante in ['mx', 'es', 'cu']:
    sub = df_test[df_test['VARIANTE'] == variante]
    dups = sub.duplicated(subset='MESSAGE').sum()
    print(f'  {variante}: {len(sub)} registros | {dups} duplicados')
print(f'Distribución IS_IRONIC: {df_test["IS_IRONIC"].value_counts().to_dict()}')
```

## 4. Extracción de variables lingüísticas

Se extraen 9 variables lingüísticas del texto original antes del preprocesamiento.
Las features son idénticas para las tres variantes de texto (normal, stem, lemma)
porque se calculan una sola vez sobre el texto sin procesar.

(González-Ibáñez et al., 2011; Joshi et al., 2015, 2016; Šandor & Bagić Babac, 2023; RAE, 2011)

```python
def quitar_tildes(texto):
    """Convierte caracteres acentuados a su forma base para capturar
    variantes de escritura informal (jamás→jamas, ningún→ningun)."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

# Lista de negaciones en español
# Fuente: RAE y Asociación de Academias de la Lengua Española (2011).
# Nueva gramática básica de la lengua española.
NEGACIONES = {
    'no', 'nunca', 'jamas', 'tampoco',
    'nada', 'nadie',
    'ningun', 'ninguno', 'ninguna', 'ningunos', 'ningunas',
    'ni', 'sino', 'sin'
}

FEATURE_COLS = ['n_exc', 'n_int', 'n_may', 'n_emo', 'n_ris',
                'n_neg', 'n_elo', 'n_com', 'n_pun']

def extraer_features_linguisticos(texto):
    """Extrae 9 variables lingüísticas como conteos sobre el texto original."""
    texto_str  = str(texto)
    texto_norm = quitar_tildes(texto_str.lower())
    palabras   = re.findall(r'\b\w+\b', texto_norm)

    # 1. Exclamaciones (González-Ibáñez et al., 2011; Joshi et al., 2015)
    n_exc = texto_str.count('!') + texto_str.count('¡')
    # 2. Interrogaciones (González-Ibáñez et al., 2011; Joshi et al., 2015)
    n_int = texto_str.count('?') + texto_str.count('¿')
    # 3. Mayúsculas enfáticas: 3+ chars (Joshi et al., 2015; Šandor & Bagić Babac, 2023)
    n_may = len(re.findall(r'\b[A-ZÁÉÍÓÚÜÑ]{3,}\b', texto_str))
    # 4. Emojis (González-Ibáñez et al., 2011; Joshi et al., 2015)
    n_emo = sum(1 for c in texto_str if c in emoji.EMOJI_DATA)
    # 5. Risas (González-Ibáñez et al., 2011)
    n_ris = len(re.findall(
        r'\b(ja+ja+|je+je+|ji+ji+|ha+ha+|xd+|lol+|jsjs)\b', texto_norm))
    # 6. Negaciones (González-Ibáñez et al., 2011; RAE, 2011)
    n_neg = sum(1 for p in palabras if p in NEGACIONES)
    # 7. Elongación: mismo carácter 3+ veces (Šandor & Bagić Babac, 2023; Joshi et al., 2016)
    n_elo = len(re.findall(r'(.)\1{2,}', texto_norm))
    # 8. Comillas (González-Ibáñez et al., 2011; Šandor & Bagić Babac, 2023)
    n_com = len(re.findall(r'["\'\u2018\u2019\u201c\u201d«»]', texto_str))
    # 9. Puntos suspensivos (Šandor & Bagić Babac, 2023; Joshi et al., 2016)
    n_pun = texto_str.count('...') + texto_str.count('\u2026')

    return [n_exc, n_int, n_may, n_emo, n_ris, n_neg, n_elo, n_com, n_pun]
```

```python
for df_name, df in [('Train', df_train), ('Test', df_test)]:
    feats = df['MESSAGE'].apply(extraer_features_linguisticos).tolist()
    for i, col in enumerate(FEATURE_COLS):
        df[col] = [f[i] for f in feats]

print(f'Features extraídos: {FEATURE_COLS}')
print('\n=== Medias features — Train ===')
print(df_train[FEATURE_COLS].mean().round(3))
print('\n=== Medias features — Test ===')
print(df_test[FEATURE_COLS].mean().round(3))
```

## 5. Preprocesamiento base
Genera MESSAGE_CLEAN: texto limpio sin puntuación, en minúsculas,
con emojis convertidos a texto y caracteres repetidos normalizados.
Este texto es la base para las tres variantes de vectorización.

```python
def preprocesar(texto):
    """Limpieza de texto: URLs, mentions, emojis a texto,
    normalización de elongación, eliminación de puntuación, minúsculas."""
    texto = re.sub(r'http\S+|www\S+', '[URL]', texto)
    texto = re.sub(r'@\w+', '[USER]', texto)
    texto = emoji.demojize(texto, language='es')
    texto = re.sub(r'(.)\1{2,}', r'\1\1', texto)
    texto = re.sub(r'[^\w\s\[\]áéíóúüñÁÉÍÓÚÜÑ]', ' ', texto)
    texto = texto.lower()
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

df_train['MESSAGE_CLEAN'] = df_train['MESSAGE'].apply(preprocesar)
df_test['MESSAGE_CLEAN']  = df_test['MESSAGE'].apply(preprocesar)
print(f'Preprocesamiento base aplicado — Train: {df_train.shape} | Test: {df_test.shape}')

for variante in ['mx', 'es', 'cu']:
    ejemplo = df_train[df_train['VARIANTE'] == variante].iloc[0]
    print(f'\n[{variante.upper()}]')
    print(f'ORIGINAL: {ejemplo["MESSAGE"]}')
    print(f'LIMPIO:   {ejemplo["MESSAGE_CLEAN"]}')
```

## 6. Stemming
Aplica SnowballStemmer sobre MESSAGE_CLEAN para obtener MESSAGE_STEM.

```python
stemmer = SnowballStemmer('spanish')

def aplicar_stemming(texto):
    return ' '.join(stemmer.stem(p) for p in str(texto).split())

print('Aplicando stemming...')
df_train['MESSAGE_STEM'] = df_train['MESSAGE_CLEAN'].apply(aplicar_stemming)
df_test['MESSAGE_STEM']  = df_test['MESSAGE_CLEAN'].apply(aplicar_stemming)

print('Ejemplos stem (clean → stem):')
for i in range(3):
    clean = df_train['MESSAGE_CLEAN'].iloc[i]
    stem  = df_train['MESSAGE_STEM'].iloc[i]
    print(f'  CLEAN: {clean[:70]}')
    print(f'  STEM:  {stem[:70]}')
    print()
```

## 7. Lematización
Aplica lematización con spaCy (es_core_news_sm) sobre MESSAGE_CLEAN
para obtener MESSAGE_LEMMA.

Si no tienes el modelo instalado ejecuta:
`!python -m spacy download es_core_news_sm`

```python
print('Cargando modelo spaCy...')
nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
print('Modelo cargado.')

def lematizar_corpus(textos, batch_size=256):
    lemas = []
    for doc in tqdm(nlp.pipe(textos, batch_size=batch_size),
                    total=len(textos), desc='Lematizando'):
        lemas.append(' '.join(t.lemma_ for t in doc if not t.is_space))
    return lemas

print('\nAplicando lematización a train...')
df_train['MESSAGE_LEMMA'] = lematizar_corpus(df_train['MESSAGE_CLEAN'].tolist())

print('Aplicando lematización a test...')
df_test['MESSAGE_LEMMA'] = lematizar_corpus(df_test['MESSAGE_CLEAN'].tolist())

print('\nEjemplos lemma (clean → lemma):')
for i in range(3):
    clean = df_train['MESSAGE_CLEAN'].iloc[i]
    lemma = df_train['MESSAGE_LEMMA'].iloc[i]
    print(f'  CLEAN: {clean[:70]}')
    print(f'  LEMMA: {lemma[:70]}')
    print()
```

## 8. Exportación por variante dialectal

Se exportan 18 archivos: 3 tipos de texto × 3 variantes × 2 splits (train/test).
Todos comparten las mismas 9 features lingüísticas, solo varía MESSAGE_CLEAN.

En 02_ml_modelos.ipynb el parámetro `PREP` selecciona el tipo de texto.

```python
def exportar_por_variante(df_tr, df_te, col_texto, sufijo):
    """
    Exporta train y test por variante dialectal para un tipo de preprocesamiento.
    Copia el texto indicado en MESSAGE_CLEAN para mantener compatibilidad
    con 02_ml_modelos.ipynb.
    """
    cols_base = ['ID', 'TOPIC', 'IS_IRONIC', 'MESSAGE', 'VARIANTE'] + FEATURE_COLS

    for split, df in [('train', df_tr), ('test', df_te)]:
        df_out = df[cols_base].copy()
        df_out['MESSAGE_CLEAN'] = df[col_texto]

        for cod in ['mx', 'es', 'cu']:
            df_out[df_out['VARIANTE'] == cod]\
                .reset_index(drop=True)\
                .to_csv(f'../data/{split}_clean{sufijo}_{cod}.csv', index=False)

    tipo = sufijo if sufijo else '(normal)'
    print(f'  {tipo:8} → train_clean{sufijo}_mx/es/cu.csv | test_clean{sufijo}_mx/es/cu.csv')

print('Exportando datasets por variante dialectal...\n')
exportar_por_variante(df_train, df_test, 'MESSAGE_CLEAN', '')
exportar_por_variante(df_train, df_test, 'MESSAGE_STEM',  '_stem')
exportar_por_variante(df_train, df_test, 'MESSAGE_LEMMA', '_lemma')
print(f'\n✓ 18 archivos generados en ../data/')
```

## 9. Resumen de archivos generados
```
data/
├── train_clean_mx.csv       test_clean_mx.csv       ← normal
├── train_clean_es.csv       test_clean_es.csv
├── train_clean_cu.csv       test_clean_cu.csv
├── train_clean_stem_mx.csv  test_clean_stem_mx.csv  ← stemming
├── train_clean_stem_es.csv  test_clean_stem_es.csv
├── train_clean_stem_cu.csv  test_clean_stem_cu.csv
├── train_clean_lemma_mx.csv test_clean_lemma_mx.csv ← lematización
├── train_clean_lemma_es.csv test_clean_lemma_es.csv
└── train_clean_lemma_cu.csv test_clean_lemma_cu.csv
```
El notebook `02_ml_modelos.ipynb` itera automáticamente sobre los tres tipos de preprocesamiento.