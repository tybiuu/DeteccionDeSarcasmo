# Fase 1 - Carga y Preprocesamiento del Dataset IroSvA
Se aplica la secuencia de preprocesamiento común a todos los modelos y se extraen las variables lingüísticas como features numéricos. Se utiliza la partición oficial train/test del dataset (Ortega-Bueno et al., 2019).

## 1. Importación de librerías

```python
import pandas as pd
import re
import emoji
```

## 2. Carga y unión del conjunto de entrenamiento
Se combinan las tres variantes dialectales (México, España, Cuba) y se eliminan los 3 duplicados identificados en la fase de exploración.

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
print(f"Train: {antes} → {len(df_train)} registros ({antes - len(df_train)} duplicados eliminados)")
```

## 3. Carga y combinación del conjunto de test
Los archivos de test del dataset IroSvA separan el texto de las etiquetas reales en dos archivos distintos. Se combinan por ID antes de unir las variantes.

```python
def cargar_test(variante, nombre_variante):
    df_texto = pd.read_csv(f'../data/irosva.{variante} - irosva.{variante}.test.csv')
    df_truth = pd.read_csv(f'../data/irosva.{variante}.test.truth.csv')
    # El archivo de texto tiene IS_IRONIC = '?' — se reemplaza con etiquetas reales
    df_texto = df_texto.drop(columns=['IS_IRONIC'])
    df = df_texto.merge(df_truth[['ID', 'IS_IRONIC']], on='ID')
    df['VARIANTE'] = nombre_variante
    return df

df_test = pd.concat([
    cargar_test('mx', 'mx'),
    cargar_test('es', 'es'),
    cargar_test('cu', 'cu')
], ignore_index=True)

print(f"Test: {len(df_test)} registros")
print(f"Distribución IS_IRONIC: {df_test['IS_IRONIC'].value_counts().to_dict()}")
```

## 4. Extracción de variables lingüísticas
Se extraen 7 variables lingüísticas del texto original antes del preprocesamiento, para capturar emojis y patrones en su forma natural. Estas variables se incorporan como features numéricos adicionales a los modelos de ML (González-Ibáñez et al., 2011; Joshi et al., 2015; Ortega-Bueno et al., 2022).

```python
# Lexicon para negaciones
NEGACIONES = {'no','nunca','jamás','ni','tampoco','nada','nadie','ningún','sin'}

def extraer_features_linguisticos(texto):
    texto_str   = str(texto)
    texto_lower = texto_str.lower()
    palabras    = re.findall(r'\b\w+\b', texto_lower)
    return [
        texto_str.count('!'),                                                                      # exclamaciones
        texto_str.count('?'),                                                                      # interrogaciones
        len(re.findall(r'\b[A-ZÁÉÍÓÚÜÑ]{3,}\b', texto_str)),                              # mayúsculas enfáticas
        sum(1 for c in texto_str if c in emoji.EMOJI_DATA),                                       # emojis
        len(re.findall(r'\b(ja+ja+|je+je+|ji+ji+|ha+ha+|xs+|xd+|lol+|jsjs)\b', texto_lower)),  # risas
        sum(1 for p in palabras if p in NEGACIONES)                                               # negaciones
    ]

FEATURE_COLS = ['n_exc', 'n_int', 'n_may', 'n_emo', 'n_ris', 'n_neg']

for df in [df_train, df_test]:
    feats = df['MESSAGE'].apply(extraer_features_linguisticos).tolist()
    for i, col in enumerate(FEATURE_COLS):
        df[col] = [f[i] for f in feats]

print(f"Features extraídos: {FEATURE_COLS}")
print(df_train[FEATURE_COLS].describe().round(2))

# Verificación: estadísticas descriptivas de features extraídos
print("=== Estadísticas features lingüísticos — Train ===")
print(df_train[['n_exc','n_int','n_may','n_emo','n_ris','n_neg']].mean().round(3))

print("\n=== Estadísticas features lingüísticos — Test ===")
print(df_test[['n_exc','n_int','n_may','n_emo','n_ris','n_neg']].mean().round(3))
```

## 5. Función de preprocesamiento
Se conservan mayúsculas enfáticas y puntuación expresiva como señales pragmáticas de ironía (Ortega-Bueno et al., 2022). Los emojis se convierten a descripciones textuales en español para preservar su carga semántica (Qin et al., 2025). Las menciones y URLs se reemplazan por tokens neutros para evitar que el modelo aprenda identidades específicas.

```python
def preprocesar(texto):
    # 1. Reemplazar URLs por token neutro
    texto = re.sub(r'http\S+|www\S+', '[URL]', texto)
    # 2. Reemplazar menciones @usuario por token neutro
    texto = re.sub(r'@\w+', '[USER]', texto)
    # 3. Convertir emojis a descripción textual en español (Qin et al., 2025)
    texto = emoji.demojize(texto, language='es')
    # 4. Normalizar caracteres repetidos: máx. 2 ocurrencias (Ortega-Bueno et al., 2022)
    texto = re.sub(r'(.)\1{2,}', r'\1\1', texto)
    # 5. Conservar puntuación enfática (¿ ¡ ! ?) como señal pragmática de ironía
    texto = re.sub(r'[^\w\s\!\?\.\,\;\:\#\[\]áéíóúüñÁÉÍÓÚÜÑ¿¡]', ' ', texto)
    # 6. Eliminar espacios múltiples
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto
```

## 6. Aplicación del preprocesamiento y verificación con ejemplos

```python
df_train['MESSAGE_CLEAN'] = df_train['MESSAGE'].apply(preprocesar)
df_test['MESSAGE_CLEAN']  = df_test['MESSAGE'].apply(preprocesar)
print(f"Preprocesamiento aplicado — Train: {df_train.shape} | Test: {df_test.shape}")

# Verificación con un ejemplo por variante
for variante in ['mx', 'es', 'cu']:
    ejemplo = df_train[df_train['VARIANTE'] == variante].iloc[0]
    print(f"\n[{variante.upper()}]")
    print(f"ORIGINAL: {ejemplo['MESSAGE']}")
    print(f"LIMPIO:   {ejemplo['MESSAGE_CLEAN']}")
```

## 7. Exportación de datos preprocesados

```python
df_train.to_csv('../data/train_clean.csv', index=False)
df_test.to_csv('../data/test_clean.csv', index=False)
print("Guardado: train_clean.csv | test_clean.csv")
print(f"Columnas exportadas: {df_train.columns.tolist()}")
```
