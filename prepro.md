# Fase 1 - Carga y Preprocesamiento del Dataset IroSvA
Se aplica la secuencia de preprocesamiento común a todos los modelos y se extraen
las 9 variables lingüísticas como features numéricos. Se utiliza la partición oficial
train/test del dataset (Ortega-Bueno et al., 2019).
Se generan archivos para tres variantes de texto: normal, con stemming y con lematización,
tanto combinados (3 variantes dialectales) como por variante individual.

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
print(f'Registros por variante: {df_train["VARIANTE"].value_counts().to_dict()}')
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

dups_test = df_test.duplicated(subset='MESSAGE').sum()
print(f'Test: {len(df_test)} registros | {dups_test} duplicados en MESSAGE')
print(f'Distribución IS_IRONIC: {df_test["IS_IRONIC"].value_counts().to_dict()}')
print(f'Registros por variante: {df_test["VARIANTE"].value_counts().to_dict()}')
```

## 4. Extracción de variables lingüísticas

Se extraen **9 variables lingüísticas** del texto original **antes del preprocesamiento**.
Las features son idénticas para las tres variantes de texto (normal, stem, lemma)
porque se calculan una sola vez sobre el texto original.

(González-Ibáñez et al., 2011; Joshi et al., 2015, 2017; Šandor & Bagić Babac, 2023; RAE, 2011)

```python
def quitar_tildes(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )

NEGACIONES = {
    'no', 'nunca', 'jamas', 'tampoco',
    'nada', 'nadie',
    'ningun', 'ninguno', 'ninguna', 'ningunos', 'ningunas',
    'ni', 'sino', 'sin'
}

def extraer_features_linguisticos(texto):
    texto_str  = str(texto)
    texto_norm = quitar_tildes(texto_str.lower())
    palabras   = re.findall(r'\b\w+\b', texto_norm)
    n_exc = texto_str.count('!') + texto_str.count('¡')
    n_int = texto_str.count('?') + texto_str.count('¿')
    n_may = len(re.findall(r'\b[A-ZÁÉÍÓÚÜÑ]{3,}\b', texto_str))
    n_emo = sum(1 for c in texto_str if c in emoji.EMOJI_DATA)
    n_ris = len(re.findall(r'\b(ja+ja+|je+je+|ji+ji+|ha+ha+|xs+|xd+|lol+|jsjs)\b', texto_norm))
    n_neg = sum(1 for p in palabras if p in NEGACIONES)
    n_elo = len(re.findall(r'(.)\1{2,}', texto_norm))
    n_com = len(re.findall(r'["\'\u2018\u2019\u201c\u201d«»]', texto_str))
    n_pun = texto_str.count('...') + texto_str.count('\u2026')
    return [n_exc, n_int, n_may, n_emo, n_ris, n_neg, n_elo, n_com, n_pun]

FEATURE_COLS = ['n_exc','n_int','n_may','n_emo','n_ris','n_neg','n_elo','n_com','n_pun']

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

## 5. Preprocesamiento base
Genera MESSAGE_CLEAN: texto limpio sin puntuación, en minúsculas,
con emojis convertidos a texto y caracteres repetidos normalizados.
Este texto es la base para las tres variantes de vectorización.

```python
def preprocesar(texto):
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
(Evaluación experimental a petición del asesor)

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
(Evaluación experimental a petición del asesor)

Si no tienes el modelo instalado ejecuta:
`!python -m spacy download es_core_news_sm`

```python
print('Cargando modelo spaCy...')
nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
print('Modelo cargado.')

def lematizar_corpus(textos, batch_size=256):
    lemas = []
    for doc in tqdm(nlp.pipe(textos, batch_size=batch_size), total=len(textos), desc='Lematizando'):
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

## 8. Exportación de todos los datasets

Se exportan 24 archivos: 3 tipos de texto × 4 datasets × 2 splits (train/test).
Todos comparten las mismas 9 features lingüísticas, solo varía MESSAGE_CLEAN.

En 02_ml_modelos.ipynb se elige el tipo de texto con el parámetro `PREP`.

```python
def exportar_variante(df_tr, df_te, col_texto, sufijo):
    """
    Exporta train y test para un tipo de preprocesamiento.
    Copia el texto indicado en MESSAGE_CLEAN para mantener
    compatibilidad con 02_ml_modelos.ipynb.
    """
    cols_base = ['ID','TOPIC','IS_IRONIC','MESSAGE','VARIANTE'] + FEATURE_COLS

    for split, df in [('train', df_tr), ('test', df_te)]:
        df_out = df[cols_base].copy()
        df_out['MESSAGE_CLEAN'] = df[col_texto]

        # Combinado
        df_out.to_csv(f'../data/{split}_clean{sufijo}.csv', index=False)

        # Por variante
        for cod in ['mx', 'es', 'cu']:
            df_out[df_out['VARIANTE'] == cod]\
                .reset_index(drop=True)\
                .to_csv(f'../data/{split}_clean{sufijo}_{cod}.csv', index=False)

    tipo = sufijo if sufijo else '(normal)'
    print(f'  {tipo:8} → train_clean{sufijo}.csv | train_clean{sufijo}_mx/es/cu.csv')
    print(f'  {"":8}   test_clean{sufijo}.csv  | test_clean{sufijo}_mx/es/cu.csv')

print('Exportando datasets...\n')
exportar_variante(df_train, df_test, 'MESSAGE_CLEAN', '')
exportar_variante(df_train, df_test, 'MESSAGE_STEM',  '_stem')
exportar_variante(df_train, df_test, 'MESSAGE_LEMMA', '_lemma')
print('\n✓ 24 archivos generados en ../data/')
```

## 9. Resumen de archivos generados
```
data/
├── train_clean.csv          test_clean.csv          ← normal
├── train_clean_mx/es/cu     test_clean_mx/es/cu
├── train_clean_stem.csv     test_clean_stem.csv     ← stemming
├── train_clean_stem_mx/es/cu  test_clean_stem_mx/es/cu
├── train_clean_lemma.csv    test_clean_lemma.csv    ← lematización
└── train_clean_lemma_mx/es/cu test_clean_lemma_mx/es/cu
```
En 02_ml_modelos.ipynb cambiar: `PREP = 'normal'` | `'stem'` | `'lemma'`