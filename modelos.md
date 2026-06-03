# Fase 2 - Modelos ML: TF-IDF y GloVe
Implementación y evaluación de LR, RF y SVM con dos representaciones textuales:
TF-IDF de palabras y GloVe SBWC en español. Se evalúa sobre el corpus combinado
(3 variantes) y por variante individual. GridSearchCV con Stratified K-Fold (5 folds)
optimizando F1-Macro (Bhattacharjee et al., 2023; Ortega-Bueno et al., 2022).

**Parámetro `PREP`** (sección 2): controla qué variante de preprocesamiento se usa.
- `'normal'`  → texto limpio base (TF-IDF + GloVe)
- `'stem'`    → texto con stemming (solo TF-IDF; GloVe omitido por OOV=100%)
- `'lemma'`   → texto lematizado  (solo TF-IDF; GloVe omitido por OOV=100%)

## 1. Importación de librerías

```python
import pandas as pd
import numpy as np
import gzip, shutil, os, time
import joblib

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

# gensim no se importa: se usa loader propio para GloVe

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    f1_score, accuracy_score,
    precision_score, recall_score, classification_report
)
```

## 2. Configuración

**Cambiar `PREP` para seleccionar la variante de preprocesamiento:**
- `'normal'` → texto limpio base
- `'stem'`   → texto con stemming
- `'lemma'`  → texto lematizado

```python
# ══════════════════════════════════════════════════════
# PARÁMETRO PRINCIPAL — cambiar aquí para cada experimento
PREP = 'stem'   # opciones: 'normal' | 'stem' | 'lemma'
# ══════════════════════════════════════════════════════

SUFIJO = '' if PREP == 'normal' else f'_{PREP}'

DATA_DIR   = '../data'
MODELS_DIR = f'../data/modelos/{PREP}'   # carpeta separada por tipo de prep
GLOVE_GZ   = '../glove-sbwc.i25.vec.gz'
GLOVE_VEC  = '../glove-sbwc.i25.vec'

os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = ['n_exc','n_int','n_may','n_emo','n_ris',
                'n_neg','n_elo','n_com','n_pun']

STOP_WORDS = stopwords.words('spanish')
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f'Preprocesamiento seleccionado : {PREP}')
print(f'Sufijo de archivos            : "{SUFIJO}" (vacío = normal)')
print(f'Carpeta de modelos            : {MODELS_DIR}')
print(f'Features lingüísticos         : {FEATURE_COLS}')
```

## 3. Carga de datasets

```python
df_train    = pd.read_csv(f'{DATA_DIR}/train_clean{SUFIJO}.csv')
df_test     = pd.read_csv(f'{DATA_DIR}/test_clean{SUFIJO}.csv')
df_train_mx = pd.read_csv(f'{DATA_DIR}/train_clean{SUFIJO}_mx.csv')
df_test_mx  = pd.read_csv(f'{DATA_DIR}/test_clean{SUFIJO}_mx.csv')
df_train_es = pd.read_csv(f'{DATA_DIR}/train_clean{SUFIJO}_es.csv')
df_test_es  = pd.read_csv(f'{DATA_DIR}/test_clean{SUFIJO}_es.csv')
df_train_cu = pd.read_csv(f'{DATA_DIR}/train_clean{SUFIJO}_cu.csv')
df_test_cu  = pd.read_csv(f'{DATA_DIR}/test_clean{SUFIJO}_cu.csv')

DATASETS = {
    'combinado': (df_train,    df_test),
    'mx':        (df_train_mx, df_test_mx),
    'es':        (df_train_es, df_test_es),
    'cu':        (df_train_cu, df_test_cu),
}

print(f'Datasets cargados ({PREP}):')
for nombre, (tr, te) in DATASETS.items():
    print(f'  {nombre:12} train={len(tr):,} | test={len(te):,}')

# Verificar ejemplo de texto
print(f'\nEjemplo MESSAGE_CLEAN ({PREP}):')
print(df_train["MESSAGE_CLEAN"].iloc[0])
```

## 4. Preparación de vectores GloVe
Solo se carga para PREP='normal'. Con stemming o lematización
el vocabulario GloVe tiene OOV=100% (GloVe fue entrenado en palabras completas).

(Pennington et al., 2014; Cardellino, 2016)

```python
glove = None

if PREP != 'normal':
    print(f'GloVe omitido para PREP="{PREP}": los vectores GloVe están '
          f'entrenados en palabras completas y son incompatibles con '
          f'{"stemming" if PREP=="stem" else "lematización"}.')
else:
    # Descomprimir si no existe
    if not os.path.exists(GLOVE_VEC):
        print('Descomprimiendo... (puede tardar ~1 min)')
        with gzip.open(GLOVE_GZ, 'rb') as f_in:
            with open(GLOVE_VEC, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print('Descompresión completada.')
    else:
        print('Archivo .vec ya existe.')

    class GloVeModel:
        def __init__(self, filepath):
            self.key_to_index = {}
            vectors_list = []
            expected_dim = None
            print('Cargando vectores GloVe...')
            t0 = time.time()
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i % 100000 == 0 and i > 0:
                        print(f'  {i:,} vectores...', end='\r')
                    parts = line.rstrip().split(' ')
                    if len(parts) < 2:
                        continue
                    # Saltar cabecera formato Word2Vec ("855380 300")
                    if len(parts) == 2:
                        try:
                            int(parts[0]); int(parts[1])
                            print(f'Cabecera detectada: {parts[0]} palabras, {parts[1]} dims')
                            continue
                        except ValueError:
                            pass
                    word = parts[0]
                    try:
                        vector = np.array(parts[1:], dtype=np.float32)
                    except ValueError:
                        continue
                    if expected_dim is None:
                        expected_dim = len(vector)
                    if len(vector) != expected_dim:
                        continue
                    self.key_to_index[word] = len(vectors_list)
                    vectors_list.append(vector)
            self.vectors     = np.vstack(vectors_list)
            self.vector_size = self.vectors.shape[1]
            print(f'\nListo en {time.time()-t0:.1f}s | '
                  f'{len(self.key_to_index):,} vectores | '
                  f'{self.vector_size} dimensiones')

        def __contains__(self, word):
            return word in self.key_to_index

        def __getitem__(self, word):
            return self.vectors[self.key_to_index[word]]

    glove = GloVeModel(GLOVE_VEC)

    # Diagnóstico
    _, oov = (lambda textos, modelo: (
        None,
        sum(1 for t in textos for w in str(t).split() if w not in modelo.key_to_index) /
        max(sum(len(str(t).split()) for t in textos), 1) * 100
    ))(df_train['MESSAGE_CLEAN'], glove)
    print(f'Tasa OOV en train combinado: {oov:.1f}%')
```

```python
def vectorizar_glove(textos, modelo, dim=300):
    vectores = []
    oov_total, tokens_total = 0, 0
    for texto in textos:
        palabras = str(texto).split()
        tokens_total += len(palabras)
        vecs = [modelo[w] for w in palabras if w in modelo.key_to_index]
        oov_total += len(palabras) - len(vecs)
        vectores.append(np.mean(vecs, axis=0) if vecs else np.zeros(dim))
    oov_rate = oov_total / max(tokens_total, 1) * 100
    return np.array(vectores), oov_rate
```

## 5. Definición de modelos y grids de hiperparámetros

```python
def build_tfidf_prep():
    return ColumnTransformer([
        ('tfidf_word', TfidfVectorizer(
            min_df=2,
            max_df=0.95,
            max_features=None  # vocabulario completo: ~8k normal, ~5k stem, ~6k lemma
        ), 'MESSAGE_CLEAN'),
        ('ling', 'passthrough', FEATURE_COLS)
    ])

def build_tfidf_pipelines():
    return {
        'LR':  Pipeline([('prep', build_tfidf_prep()),
                         ('clf',  LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced'))]),
        'RF':  Pipeline([('prep', build_tfidf_prep()),
                         ('clf',  RandomForestClassifier(random_state=42, class_weight='balanced'))]),
        'SVM': Pipeline([('prep', build_tfidf_prep()),
                         ('clf',  LinearSVC(random_state=42, class_weight='balanced', max_iter=100000))])
    }

GRIDS_TFIDF = {
    'LR': {
        'prep__tfidf_word__ngram_range': [(1,1), (1,2)],
        'clf__C':                        [0.1, 1.0, 10],
        'clf__solver':                   ['liblinear']
    },
    'RF':  {
        'prep__tfidf_word__ngram_range': [(1,1), (1,2)],
        'clf__n_estimators':             [100, 200, 300],
        'clf__max_depth':                [None, 10, 20],
        'clf__min_samples_split':        [2, 5]
    },
    'SVM': {
        'prep__tfidf_word__ngram_range': [(1,1), (1,2)],
        'clf__C':                        [0.01, 0.1, 1.0, 10]
    }
}

def build_glove_pipelines():
    return {
        'LR':  Pipeline([('scaler', StandardScaler()),
                         ('clf',    LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced'))]),
        'RF':  Pipeline([('scaler', StandardScaler()),
                         ('clf',    RandomForestClassifier(random_state=42, class_weight='balanced'))]),
        'SVM': Pipeline([('scaler', StandardScaler()),
                         ('clf',    LinearSVC(random_state=42, class_weight='balanced', max_iter=100000))])
    }

GRIDS_GLOVE = {
    'LR': {
        'clf__C':        [0.1, 1.0, 10],
        'clf__solver':   ['liblinear']
    },
    'RF': {
        'clf__n_estimators':      [100, 200, 300],
        'clf__max_depth':         [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    },
    'SVM': {'clf__C': [0.01, 0.1, 1.0, 10]}
}

print('Pipelines y grids definidos.')
```

## 6. Función principal de experimentos

```python
RESULTADOS = []

def run_experimento(dataset_nombre, df_tr, df_te,
                    repr_nombre, pipelines, grids,
                    X_tr, y_tr, X_te, y_te):
    print(f'\n{"="*60}')
    print(f'  PREP={PREP} | Dataset={dataset_nombre.upper()} | Repr={repr_nombre}')
    print(f'{"="*60}')

    for modelo_nombre in ['LR', 'RF', 'SVM']:
        pipeline = pipelines[modelo_nombre]
        grid     = grids[modelo_nombre]

        n_cand = 1
        for v in grid.values():
            n_cand *= len(v)
        print(f'\n  → {modelo_nombre} | {n_cand} candidatos...', end=' ')
        t0 = time.time()

        search = GridSearchCV(
            pipeline, grid,
            cv=CV, scoring='f1_macro',
            n_jobs=-1, verbose=0
        )
        search.fit(X_tr, y_tr)

        y_pred = search.best_estimator_.predict(X_te)
        acc    = accuracy_score(y_te, y_pred)
        f1m    = f1_score(y_te, y_pred, average='macro')
        p_ir   = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
        r_ir   = recall_score(y_te, y_pred, pos_label=1, zero_division=0)
        f1_ir  = f1_score(y_te, y_pred, pos_label=1, zero_division=0)

        print(f'F1-Macro CV={search.best_score_:.4f} | '
              f'F1-Macro test={f1m:.4f} | '
              f'tiempo={time.time()-t0:.1f}s')
        print(f'     Mejores params: {search.best_params_}')

        RESULTADOS.append({
            'prep':        PREP,
            'dataset':     dataset_nombre,
            'repr':        repr_nombre,
            'modelo':      modelo_nombre,
            'f1_macro_cv': round(search.best_score_, 4),
            'accuracy':    round(acc, 4),
            'f1_macro':    round(f1m, 4),
            'p_ironico':   round(p_ir, 4),
            'r_ironico':   round(r_ir, 4),
            'f1_ironico':  round(f1_ir, 4),
            'best_params': search.best_params_,
            'estimator':   search.best_estimator_
        })

        fname = f'{MODELS_DIR}/ml_tf-idf_{modelo_nombre.lower()}_{dataset_nombre}.pkl'
        if repr_nombre == 'GloVe':
            fname = f'{MODELS_DIR}/ml_glove_{modelo_nombre.lower()}_{dataset_nombre}.pkl'
        joblib.dump(search.best_estimator_, fname)

print('Función de experimentos lista.')
```

## 7. Experimentos TF-IDF

```python
for ds_nombre, (df_tr, df_te) in DATASETS.items():
    X_tr = df_tr[['MESSAGE_CLEAN'] + FEATURE_COLS]
    y_tr = df_tr['IS_IRONIC'].values
    X_te = df_te[['MESSAGE_CLEAN'] + FEATURE_COLS]
    y_te = df_te['IS_IRONIC'].values

    run_experimento(
        dataset_nombre=ds_nombre,
        df_tr=df_tr, df_te=df_te,
        repr_nombre='TF-IDF',
        pipelines=build_tfidf_pipelines(),
        grids=GRIDS_TFIDF,
        X_tr=X_tr, y_tr=y_tr,
        X_te=X_te, y_te=y_te
    )
```

## 8. Experimentos GloVe
Solo se ejecuta para PREP='normal'. Para stem y lemma GloVe se omite
porque los tokens stemmizados/lematizados no están en el vocabulario GloVe
(entrenado en palabras completas del español).

```python
if PREP != 'normal':
    print(f'Experimentos GloVe omitidos para PREP="{PREP}".')
    print('GloVe es incompatible con texto stemmizado/lematizado (OOV=100%).')
else:
    for ds_nombre, (df_tr, df_te) in DATASETS.items():
        print(f'\nVectorizando GloVe: {ds_nombre}...')

        glove_tr, oov_tr = vectorizar_glove(df_tr['MESSAGE_CLEAN'], glove)
        glove_te, oov_te = vectorizar_glove(df_te['MESSAGE_CLEAN'], glove)

        print(f'  OOV train={oov_tr:.1f}% | OOV test={oov_te:.1f}%')

        X_tr = np.hstack([glove_tr, df_tr[FEATURE_COLS].values.astype(float)])
        y_tr = df_tr['IS_IRONIC'].values
        X_te = np.hstack([glove_te, df_te[FEATURE_COLS].values.astype(float)])
        y_te = df_te['IS_IRONIC'].values

        run_experimento(
            dataset_nombre=ds_nombre,
            df_tr=df_tr, df_te=df_te,
            repr_nombre='GloVe',
            pipelines=build_glove_pipelines(),
            grids=GRIDS_GLOVE,
            X_tr=X_tr, y_tr=y_tr,
            X_te=X_te, y_te=y_te
        )
```

## 9. Tabla comparativa y exportación

```python
df_res = pd.DataFrame([{
    'Prep':        r['prep'],
    'Dataset':     r['dataset'],
    'Repr':        r['repr'],
    'Modelo':      r['modelo'],
    'F1-Macro CV': r['f1_macro_cv'],
    'Accuracy':    r['accuracy'],
    'F1-Macro':    r['f1_macro'],
    'F1-Irónico':  r['f1_ironico'],
    'P-Irónico':   r['p_ironico'],
    'R-Irónico':   r['r_ironico'],
} for r in RESULTADOS])

print(f'TABLA COMPARATIVA — PREP={PREP}')
print(df_res.to_string(index=False))

print(f'\nMEJOR MODELO TF-IDF POR DATASET (PREP={PREP}):')
print('-' * 60)
for ds in ['combinado','mx','es','cu']:
    subset = df_res[(df_res['Dataset']==ds) & (df_res['Repr']=='TF-IDF')]
    if not subset.empty:
        mejor = subset.loc[subset['F1-Macro'].idxmax()]
        print(f'{ds:12} {mejor["Modelo"]:5} F1-Macro={mejor["F1-Macro"]:.4f}')

# Guardar CSV con sufijo del preprocesamiento
out_csv = f'{DATA_DIR}/resultados_ml_{PREP}.csv'
df_res.to_csv(out_csv, index=False)
print(f'\nGuardado: {out_csv}')
```