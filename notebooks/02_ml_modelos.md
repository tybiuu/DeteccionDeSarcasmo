# Fase 2 - Modelos ML: Regresión Logística, Random Forest y SVM
Implementación, búsqueda de hiperparámetros y evaluación final de los tres modelos de machine learning. Se combina vectorización TF-IDF de palabras y de caracteres con 6 features lingüísticos mediante ColumnTransformer. GridSearchCV con Stratified K-Fold evita data leakage y garantiza representación balanceada en cada fold (Bhattacharjee et al., 2023; Fatima et al., 2024).

## 1. Importación de librerías

```python
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    f1_score, classification_report,
    accuracy_score, precision_score, recall_score
)
import joblib

nltk.download('stopwords')
```

## 2. Carga de datos preprocesados

```python
df_train = pd.read_csv('../data/train_clean.csv')
df_test  = pd.read_csv('../data/test_clean.csv')

FEATURE_COLS = ['n_exc', 'n_int', 'n_may', 'n_emo', 'n_ris', 'n_neg']

X_train = df_train[['MESSAGE_CLEAN'] + FEATURE_COLS]
y_train = df_train['IS_IRONIC']
X_test  = df_test[['MESSAGE_CLEAN'] + FEATURE_COLS]
y_test  = df_test['IS_IRONIC']

print(f"Train: {X_train.shape} | Test: {X_test.shape}")
print(f"Distribución train: {y_train.value_counts().to_dict()}")
```

## 3. Preprocesador y pipelines
Se combina TF-IDF de palabras (10 000 términos) con TF-IDF de caracteres (5 000 términos, `analyzer='char_wb'`) y los 6 features lingüísticos. Los n-gramas de caracteres capturan patrones morfológicos e informales propios de redes sociales; los equipos con mejores resultados en IroSvA'19 combinaron n-gramas de palabras y de caracteres (Ortega-Bueno et al., 2019). Se eliminan stopwords en el vectorizador de palabras dado que TF-IDF no captura contexto gramatical (Manoleasa et al., 2022). Se aplica `class_weight='balanced'` para compensar el desbalance 2:1 del dataset (Fatima et al., 2024).

```python
stop_words_es = stopwords.words('spanish')

def build_preprocessor():
    return ColumnTransformer([
        ('tfidf_word', TfidfVectorizer(
            stop_words=stop_words_es,
            max_features=10000   # Pandey & Singh (2023); Sandor & Bagic Babac (2023)
        ), 'MESSAGE_CLEAN'),
        ('tfidf_char', TfidfVectorizer(
            max_features=5000,
            analyzer='char_wb'   # Ortega-Bueno et al. (2019)
        ), 'MESSAGE_CLEAN'),
        ('ling', 'passthrough', FEATURE_COLS)
    ])

# Pipelines
pipeline_lr = Pipeline([
    ('prep', build_preprocessor()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

pipeline_rf = Pipeline([
    ('prep', build_preprocessor()),
    ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

pipeline_svm = Pipeline([
    ('prep', build_preprocessor()),
    ('clf', LinearSVC(random_state=42, class_weight='balanced', max_iter=10000))
])

# Grillas de hiperparámetros
param_grid_lr = {
    'prep__tfidf_word__ngram_range': [(1,1), (1,2)],   # Sandor & Bagic Babac (2023); Bhattacharjee et al. (2023)
    'prep__tfidf_char__ngram_range': [(2,4), (3,5)],
    'clf__C':                        [0.1, 1.0, 10],
    'clf__solver':                   ['liblinear']
}

param_grid_rf = {
    'prep__tfidf_word__ngram_range': [(1,1), (1,2)],
    'prep__tfidf_char__ngram_range': [(2,4), (3,5)],
    'clf__n_estimators':             [100, 200],        # Pandey & Singh (2023)
    'clf__max_depth':                [None, 10, 20],    # Pandey & Singh (2023)
    'clf__min_samples_split':        [2, 5]             # Pandey & Singh (2023)
}

param_grid_svm = {
    'prep__tfidf_word__ngram_range': [(1,1), (1,2)],
    'prep__tfidf_char__ngram_range': [(2,4), (3,5)],
    'clf__C':                        [0.01, 0.1, 1.0]  # Pandey & Singh (2023); Sandor & Bagic Babac (2023)
}

print("Pipelines y grillas definidos")
```

## 4. Grid Search - Regresión Logística
StratifiedKFold con 5 folds garantiza la misma proporción de clases en cada partición, necesario ante el desbalance 2:1 (Bhattacharjee et al., 2023). La métrica de optimización es F1-Macro, consistente con la métrica principal del estudio (Sinha et al., 2023; Ortega-Bueno et al., 2022).

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('Buscando hiperparámetros para LR...')
search_lr = GridSearchCV(
    pipeline_lr, param_grid_lr,
    cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1
)
search_lr.fit(X_train, y_train)
print(f"Mejores parámetros LR: {search_lr.best_params_}")
print(f"Mejor F1-Macro LR (CV): {search_lr.best_score_:.4f}")
```

## 5. Grid Search - Random Forest

```python
print('Buscando hiperparámetros para RF...')
search_rf = GridSearchCV(
    pipeline_rf, param_grid_rf,
    cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1
)
search_rf.fit(X_train, y_train)
print(f"Mejores parámetros RF: {search_rf.best_params_}")
print(f"Mejor F1-Macro RF (CV): {search_rf.best_score_:.4f}")
```

## 6. Grid Search - SVM (LinearSVC)
SVM fue el clasificador tradicional más utilizado por los equipos participantes en IroSvA'19 para detección de ironía en español (Ortega-Bueno et al., 2019). Shelke & Wagh (2025) reportan que SVM obtuvo el mejor rendimiento entre los modelos de ML clásico en su estudio comparativo. Roy et al. (2026) lo incluyen como baseline en su conjunto exhaustivo de comparación. Los valores de C evaluados cubren el rango reportado en la literatura (Pandey & Singh, 2023; Šandor & Bagić Babac, 2023).

```python
print('Buscando hiperparámetros para SVM...')
search_svm = GridSearchCV(
    pipeline_svm, param_grid_svm,
    cv=cv, scoring='f1_macro', n_jobs=-1, verbose=1
)
search_svm.fit(X_train, y_train)
print(f"Mejores parámetros SVM: {search_svm.best_params_}")
print(f"Mejor F1-Macro SVM (CV): {search_svm.best_score_:.4f}")
```

## 7. Evaluación final en conjunto de test

```python
best_lr  = search_lr.best_estimator_
best_rf  = search_rf.best_estimator_
best_svm = search_svm.best_estimator_

y_pred_lr  = best_lr.predict(X_test)
y_pred_rf  = best_rf.predict(X_test)
y_pred_svm = best_svm.predict(X_test)

for nombre, y_pred in [
    ('Regresión Logística', y_pred_lr),
    ('Random Forest',       y_pred_rf),
    ('SVM (LinearSVC)',     y_pred_svm)
]:
    print(f'=== {nombre} ===')
    print(f"F1-Macro test: {f1_score(y_test, y_pred, average='macro'):.4f}")
    print(classification_report(y_test, y_pred, target_names=['No irónico', 'Irónico']))

# Tabla comparativa
print(f"\n{'Modelo':<22} {'Accuracy':>10} {'F1-Macro':>10} {'F1-CV':>8} {'P-Irónico':>12} {'R-Irónico':>12}")
print('-' * 78)
for nombre, y_pred, cv_score in [
    ('Reg. Logística',  y_pred_lr,  search_lr.best_score_),
    ('Random Forest',   y_pred_rf,  search_rf.best_score_),
    ('SVM (LinearSVC)', y_pred_svm, search_svm.best_score_),
]:
    acc  = accuracy_score(y_test, y_pred)
    f1m  = f1_score(y_test, y_pred, average='macro')
    p_ir = precision_score(y_test, y_pred, pos_label=1)
    r_ir = recall_score(y_test, y_pred, pos_label=1)
    print(f"{nombre:<22} {acc:>10.4f} {f1m:>10.4f} {cv_score:>8.4f} {p_ir:>12.4f} {r_ir:>12.4f}")
```

## 8. Exportación de modelos

```python
joblib.dump(best_lr,  '../data/modelo_lr.pkl')
joblib.dump(best_rf,  '../data/modelo_rf.pkl')
joblib.dump(best_svm, '../data/modelo_svm.pkl')
print('Modelos guardados: modelo_lr.pkl | modelo_rf.pkl | modelo_svm.pkl')
```
