# Documentación del Proyecto: Predicción de Riesgo de Trastorno del Sueño

**Dataset:** `sleep_health_dataset.csv` (100,000 registros, 32 variables)
**Variable objetivo:** `sleep_disorder_risk` (multiclase: Healthy / Mild / Moderate / Severe)
**Metodología:** CRISP-DM
**Notebook:** `EDA_Preprocesamiento_Sleep_Health.ipynb`

---

## 1. Comprensión del Negocio (Fase 1 CRISP-DM)

### 1.1 Problema

Los trastornos del sueño afectan la salud física, el rendimiento cognitivo y el bienestar general. La detección temprana de personas en riesgo permite intervenciones preventivas (cambios de hábitos, evaluación clínica, ajuste de turnos laborales). El objetivo de este proyecto es construir un **clasificador multiclase** que prediga el nivel de riesgo de trastorno del sueño a partir de variables fisiológicas, conductuales y contextuales.

### 1.2 Objetivos

- **Objetivo de negocio:** disponer de un modelo que clasifique correctamente individuos en cuatro niveles de riesgo (Healthy, Mild, Moderate, Severe) para apoyar decisiones de tamizaje.
- **Objetivo de minería de datos:** alcanzar un **F1-weighted ≥ 0.85** sobre el conjunto de prueba, prestando atención al desempeño en clases minoritarias (Moderate, Severe).
- **Criterios de éxito:** modelo robusto (gap CV–Test bajo), reproducible, con interpretabilidad mínima vía importancia de features.

---

## 2. Comprensión de los Datos (Fase 2 CRISP-DM)

### 2.1 Estructura General

| Característica | Valor |
|---|---|
| Filas | 100,000 |
| Columnas | 32 |
| Numéricas | 24 |
| Categóricas (object) | 7 |
| Identificador | `person_id` (descartado) |
| Valores nulos | 0 |
| Duplicados | 0 |

### 2.2 Variable Objetivo

`sleep_disorder_risk` con 4 categorías ordinales:

| Clase | Conteo | Proporción |
|---|---:|---:|
| Healthy | 54,156 | 54.16 % |
| Mild | 33,479 | 33.48 % |
| Moderate | 8,299 | 8.30 % |
| Severe | 4,066 | 4.07 % |

> **Hallazgo:** dataset **moderadamente desbalanceado**. Las clases minoritarias (Moderate, Severe) suman ~12 % y serán las más difíciles de predecir. Se aplica `class_weight='balanced'` en los modelos compatibles.

### 2.3 Variables Más Correlacionadas con el Target

Top correlaciones absolutas con el target codificado (0=Healthy … 3=Severe):

| Variable | Correlación |
|---|---:|
| cognitive_performance_score | -0.704 |
| sleep_quality_score | -0.694 |
| stress_score | +0.484 |
| sleep_duration_hrs | -0.474 |
| felt_rested | -0.387 |
| wake_episodes_per_night | +0.347 |
| sleep_latency_mins | +0.333 |
| shift_work | +0.241 |
| bmi | +0.237 |

Los signos son coherentes clínicamente: mayor calidad de sueño y rendimiento cognitivo se asocian a menor riesgo; mayor estrés, latencia y despertares se asocian a mayor riesgo.

### 2.4 Outliers

Análisis IQR sobre 9 variables numéricas clave: la presencia de outliers es **baja a moderada** y consistente con variabilidad biológica esperada (no se trata como ruido). No se realiza recorte ni winsorización para preservar señal.

---

## 3. Preparación de los Datos (Fase 3 CRISP-DM)

### 3.1 Decisiones de Preprocesamiento

| Operación | Variables | Justificación |
|---|---|---|
| Drop | `person_id` | Identificador, no aporta señal |
| Label encoding | `sleep_disorder_risk` | Mapeo ordinal Healthy(0) < Mild(1) < Moderate(2) < Severe(3) |
| StandardScaler | 23 numéricas | Algoritmos basados en distancias y gradiente requieren features comparables |
| OrdinalEncoder | `chronotype` (Morning < Neutral < Evening) | Variable con orden natural |
| OneHotEncoder | `gender, occupation, country, mental_health_condition, season, day_type` | Variables sin orden, `drop='if_binary'` |

### 3.2 Partición Train/Test

División **80/20 estratificada** por la variable objetivo:

| Conjunto | Filas | Features | Distribución target |
|---|---:|---:|---|
| Train | 80,000 | 63 | 54.2/33.5/8.3/4.1 % |
| Test | 20,000 | 63 | 54.2/33.5/8.3/4.1 % |

El número de features sube de 31 (sin id) a **63** tras One-Hot Encoding de variables nominales (principalmente `country` y `occupation` con muchas categorías).

### 3.3 Comparación StandardScaler vs MinMaxScaler

Baseline `LogisticRegression` con 5-fold CV:

| Escalador | F1-weighted | Tiempo |
|---|---:|---:|
| StandardScaler | 0.8138 ± 0.0031 | 4.6 s |
| MinMaxScaler | 0.8134 ± 0.0031 | 4.9 s |

Diferencia despreciable. Se elige **StandardScaler** por convención.

---

## 4. Modelado (Fase 4 CRISP-DM)

### 4.1 Estrategia

Dada la escala del dataset (100k filas), se aplican optimizaciones para mantener el cómputo viable sin sacrificar rigor:

1. **Muestra estratificada de 20,000 filas** del set de entrenamiento para el tuning y CV (mantiene proporciones de clase).
2. **GridSearchCV** sobre los 8 clasificadores → cada uno se evalúa con su mejor configuración.
3. **10-Fold Stratified CV** comparativa con los modelos optimizados → ranking justo.
4. **Evaluación final** en el test set completo (20,000 filas).

> **Nota metodológica:** se invirtió el orden frente al patrón "CV con defaults → GridSearch sobre top-3" porque la comparación rigurosa requiere que **todos** los modelos compitan con su mejor versión.

### 4.2 Reemplazos por Rendimiento

Frente al notebook plantilla original (Social Media Impact, 1,705 filas), se hicieron tres reemplazos:

| Original | Reemplazo | Motivo |
|---|---|---|
| `SVC(kernel='rbf')` | `LinearSVC` | SVM RBF es O(n²), inviable en 100k |
| `GradientBoostingClassifier` | `HistGradientBoostingClassifier` | 10× más rápido, mismas garantías |
| Sin `n_jobs` | `n_jobs=-1` | Aprovecha CPU multicore |

### 4.3 Resultados de GridSearchCV (5-fold, F1-weighted)

| Modelo | F1w CV5 | Mejores hiperparámetros | Tiempo |
|---|---:|---|---:|
| **HistGradientBoosting** | **0.9341** | `learning_rate=0.1, max_depth=None, max_iter=200` | 60.5 s |
| Random Forest | 0.8511 | `n_estimators=200, max_depth=16` | 9.6 s |
| Decision Tree | 0.8201 | `max_depth=12, min_samples_split=2` | 1.0 s |
| Logistic Regression | 0.8113 | `C=5.0` | 1.3 s |
| LinearSVC | 0.8047 | `C=1.0` | 1.3 s |
| AdaBoost | 0.7089 | `n_estimators=100, learning_rate=0.5` | 19.8 s |
| KNN | 0.7055 | `n_neighbors=9, weights=distance` | 4.6 s |
| Naive Bayes | 0.6926 | `var_smoothing=1e-7` | 0.2 s |

**Tiempo total GridSearch: 98.2 s** (≈1.6 min).

### 4.4 Validación Cruzada Final 10-Fold con Modelos Optimizados

| # | Modelo | Accuracy CV | Std |
|--:|---|---:|---:|
| 1 | **HistGradientBoosting** | **93.84 %** | ±0.47 |
| 2 | Random Forest | 86.00 % | ±0.69 |
| 3 | Decision Tree | 81.85 % | ±0.62 |
| 4 | LinearSVC | 81.02 % | ±1.08 |
| 5 | Logistic Regression | 80.69 % | ±0.62 |
| 6 | KNN | 72.98 % | ±0.79 |
| 7 | AdaBoost | 70.49 % | ±1.59 |
| 8 | Naive Bayes | 68.83 % | ±1.24 |

**Modelo seleccionado: `HistGradientBoosting`** con configuración óptima.

---

## 5. Evaluación (Fase 5 CRISP-DM)

### 5.1 Modelo Final en Test Set Completo (20,000 filas)

`HistGradientBoosting` re-entrenado en train completo (80k) y evaluado en test completo (20k):

| Métrica | Valor |
|---|---:|
| Accuracy | **0.9538** |
| Precision (weighted) | 0.9532 |
| Recall (weighted) | 0.9538 |
| **F1 (weighted)** | **0.9534** |

### 5.2 Classification Report Detallado

| Clase | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 1.00 | 1.00 | 1.00 | 10,831 |
| Mild | 0.94 | 0.96 | 0.95 | 6,696 |
| Moderate | 0.73 | 0.72 | 0.73 | 1,660 |
| Severe | 0.87 | 0.79 | 0.83 | 813 |
| **macro avg** | 0.89 | 0.87 | 0.88 | 20,000 |
| **weighted avg** | 0.95 | 0.95 | 0.95 | 20,000 |

### 5.3 F1 por Clase y Modelo (Test)

| Modelo | Healthy | Mild | Moderate | Severe |
|---|---:|---:|---:|---:|
| **HistGradientBoosting** | **0.999** | **0.952** | **0.727** | **0.826** |
| Random Forest | 0.949 | 0.858 | 0.626 | 0.761 |
| Decision Tree | 0.958 | 0.836 | 0.576 | 0.698 |
| LinearSVC | 0.910 | 0.746 | 0.491 | 0.696 |
| Logistic Regression | 0.906 | 0.743 | 0.548 | 0.726 |
| KNN | 0.867 | 0.649 | 0.285 | 0.458 |
| Naive Bayes | 0.852 | 0.582 | 0.307 | 0.546 |
| AdaBoost | 0.798 | 0.709 | 0.558 | 0.000 |

> **Observación:** HistGradientBoosting es el único modelo con buen desempeño en las cuatro clases. **AdaBoost colapsa en la clase Severe (F1=0.00)**, mostrando incapacidad para capturar la clase minoritaria.

### 5.4 Detección de Overfitting (CV vs Test)

El gap entre CV (sobre muestra de 20k) y test (20k) es **mínimo en todos los modelos**, lo que indica **ausencia de overfitting significativo**:

- HistGradientBoosting: CV 93.84 % → Test 95.38 % (diferencia +1.54 % a favor del test, atribuible al re-entrenamiento con train completo de 80k).
- El resto de modelos muestran diferencias menores a 2 puntos.

### 5.5 Conclusión de la Evaluación

El modelo cumple el criterio de éxito (F1-weighted ≥ 0.85) con margen amplio (0.95). Las clases minoritarias —Moderate y Severe— son las que más limitan el desempeño global, pero aún alcanzan F1 ≥ 0.73 y 0.83 respectivamente.

---

## 6. Selección de Características

Se aplicaron **7 métodos** sobre la muestra de 20k para identificar las features más predictivas:

1. **Correlación** con target ordinal
2. **ANOVA F-test** (`SelectKBest`)
3. **RFECV** con LogisticRegression (n óptimo: 48 features)
4. **Importancia RandomForest**
5. **Permutation Importance** sobre HistGradientBoosting
6. **SHAP** (TreeExplainer sobre HistGradientBoosting, 1,000 muestras)
7. **Mutual Information**

### 6.1 Features con Consenso (seleccionadas por los 7 métodos)

| Feature | n_metodos |
|---|---:|
| bmi | 7 |
| sleep_duration_hrs | 7 |
| sleep_quality_score | 7 |
| sleep_latency_mins | 7 |
| rem_percentage | 7 |
| stress_score | 7 |
| wake_episodes_per_night | 7 |
| cognitive_performance_score | 7 |
| shift_work | 6 |
| mental_health_condition_Healthy | 6 |
| mental_health_condition_Both | 6 |
| mental_health_condition_Anxiety | 5 |

### 6.2 Análisis k óptimo (SelectKBest con LogReg)

| k | F1-weighted |
|--:|---:|
| 5 | 0.6958 |
| 10 | 0.7578 |
| 15 | 0.8057 |
| 20 | 0.8085 |
| 25 | 0.8090 |
| 30 | 0.8116 |
| 40 | 0.8103 |
| 63 | 0.8117 |

> **Hallazgo:** con solo **15-20 features** (en vez de 63) se conserva ~99 % del rendimiento del modelo lineal. Hay redundancia entre las variables one-hot y las numéricas correlacionadas.

### 6.3 Conclusiones de Feature Selection

- Las variables más predictivas son **fisiológicas y conductuales del sueño** (calidad, duración, latencia, despertares, estrés) más **condición de salud mental**.
- `country` y la mayoría de categorías de `occupation` aportan poco individualmente.
- Los métodos lineales (correlación, ANOVA, MI) priorizan `sleep_quality_score` y `cognitive_performance_score`; los métodos basados en árboles (RF, HGB, SHAP) priorizan `sleep_duration_hrs`, `bmi` y `stress_score`. Esto es consistente con que los modelos no lineales aprovechan interacciones que las correlaciones marginales no capturan.

---

## 7. Análisis de Robustez: Comparaciones Sin Features Específicas

Se evaluó la dependencia del modelo respecto a features dominantes, removiendo cada una y re-entrenando los 8 modelos.

### 7.1 Sin `sleep_quality_score`

| Modelo | Con (CV%) | Sin (CV%) | Δ |
|---|---:|---:|---:|
| HistGradientBoosting | 93.84 | **93.97** | -0.13 |
| Random Forest | 86.00 | 86.58 | -0.58 |
| Decision Tree | 81.85 | 82.93 | -1.08 |
| LinearSVC | 81.02 | 80.91 | +0.10 |
| Logistic Regression | 80.68 | 80.55 | +0.14 |
| KNN | 72.98 | 71.96 | +1.03 |
| Naive Bayes | 68.82 | 67.66 | +1.17 |
| AdaBoost | 70.49 | 68.54 | +1.95 |

> **Hallazgo clave:** remover `sleep_quality_score` **no afecta a HistGradientBoosting** (incluso mejora ligeramente). Esto sugiere **redundancia** con otras features del sueño (`cognitive_performance_score`, `sleep_duration_hrs`). El modelo final es **robusto** a esta variable.

### 7.2 Sin `mental_health_condition`

| Modelo | Con (CV%) | Sin (CV%) | Δ |
|---|---:|---:|---:|
| HistGradientBoosting | 93.84 | 83.26 | **+10.58** |
| Decision Tree | 81.85 | 72.06 | +9.78 |
| Naive Bayes | 68.82 | 60.46 | +8.36 |
| Logistic Regression | 80.68 | 73.37 | +7.32 |
| Random Forest | 86.00 | 79.51 | +6.49 |
| LinearSVC | 81.02 | 74.80 | +6.22 |
| KNN | 72.98 | 70.87 | +2.11 |
| AdaBoost | 70.49 | 73.90 | -3.41 |

> **Hallazgo crítico:** la variable `mental_health_condition` es **fundamental**. Removerla causa una caída de **10.6 puntos** en HGB (de 93.84 % a 83.26 %). Sin esta variable, el modelo final pasaría de F1=0.95 a F1=0.84 en test.

### 7.3 Ablación Sistemática (Sección 10)

Comparación de F1-weighted en test al remover dummies específicas de `mental_health_condition`:

| Experimento | HGB | RF | LogReg | media_modelos |
|---|---:|---:|---:|---:|
| Baseline (todas) | 0.953 | 0.884 | 0.814 | 0.811 |
| Sin mhc_Depression | 0.952 | 0.882 | 0.814 | 0.808 |
| Sin mhc_Anxiety | 0.953 | 0.881 | 0.814 | 0.808 |
| Sin sleep_quality_score | 0.952 | 0.894 | 0.814 | 0.806 |
| Sin mhc_Both | 0.953 | 0.879 | 0.814 | 0.804 |
| Sin mhc_Healthy | 0.952 | 0.868 | 0.814 | 0.800 |
| Sin mhc_Healthy + stress | 0.901 | 0.855 | 0.805 | 0.786 |
| **Sin todas mhc** | **0.842** | 0.799 | 0.738 | 0.739 |

> **Conclusión:** la importancia de `mental_health_condition` proviene del **conjunto** de sus 4 dummies, no de una sola categoría. Quitar cualquiera individualmente apenas mueve el modelo, pero quitarlas todas a la vez es **el experimento más dañino**.

---

## 8. Conclusiones Generales

### 8.1 Modelo Final

- **Algoritmo:** `HistGradientBoostingClassifier`
- **Hiperparámetros:** `learning_rate=0.1, max_depth=None, max_iter=200`
- **Métrica principal:** F1-weighted = **0.9534** en test set de 20,000 instancias
- **Accuracy:** 95.38 %
- **Tiempo de entrenamiento (train completo 80k):** 15.6 s

### 8.2 Hallazgos Metodológicos

1. **GridSearch antes de la comparación CV** produce un ranking más justo que comparar con hiperparámetros por defecto. En este dataset, el orden del ranking se mantuvo, pero los gaps absolutos cambiaron (Decision Tree subió de ~73 % a 82 %).
2. **Muestreo estratificado de 20k** para CV es suficiente: la diferencia con el test completo es < 2 puntos en todos los modelos.
3. **Reemplazos de modelos** (`LinearSVC`, `HistGradientBoosting`) reducen el tiempo total de minutos a < 5 minutos sin perder calidad.

### 8.3 Hallazgos del Dominio

1. Las **variables fisiológicas y conductuales del sueño** son las más predictivas, como era esperable.
2. La **condición de salud mental** es **crítica**: su ausencia degrada el modelo en >10 puntos. Esto sugiere un fuerte vínculo entre salud mental y trastornos del sueño en este dataset.
3. Las **clases minoritarias (Moderate, Severe)** son intrínsecamente más difíciles. Solo modelos no lineales con balanceo adecuado las capturan bien.
4. Variables demográficas crudas (`country`, la mayoría de `occupation`) aportan poca señal individual.

### 8.4 Limitaciones

- El dataset puede ser sintético (proporciones limpias, sin nulos, sin duplicados, distribuciones suaves). Resultados pueden no transferirse 1:1 a datos reales clínicos.
- No se realizó análisis temporal ni de cohortes (ej. estabilidad por país, estación, tipo de día).
- No se usaron técnicas de reweighting avanzadas (SMOTE, Tomek Links) más allá de `class_weight='balanced'`.

### 8.5 Trabajo Futuro

- Probar **XGBoost / LightGBM / CatBoost** como ensambles alternativos.
- Aplicar **calibración de probabilidades** (Platt, isotónica) para uso en tamizaje médico.
- Explorar **feature engineering** sobre interacciones (ej. `sleep_duration × stress_score`).
- Construir un **modelo simplificado** con solo 15-20 features (consenso de feature selection) para uso en producción ligera.

---

## 9. Estructura del Notebook

| Sección | Contenido |
|---|---|
| 1 | Imports |
| 2 | Carga de datos |
| 3 | EDA (estructura, descriptivas, target, distribuciones, correlaciones, outliers) |
| 4 | Preprocesamiento (drop, encoding, split, pipeline, comparación de scalers) |
| 5 | Modelado (GridSearchCV → CV final → mejor modelo → test) |
| 6 | Evaluación detallada (métricas, CM, F1 por clase, overfitting) |
| 7 | Feature Selection (7 métodos + consenso) |
| 8 | Comparación sin `sleep_quality_score` |
| 9 | Comparación sin `mental_health_condition` |
| 10 | Ablación sistemática de dummies de mental_health_condition |

## 10. Reproducibilidad

- **Random state:** 42 en todos los splits y modelos.
- **Versión de Python:** 3.10+
- **Librerías clave:** scikit-learn ≥ 1.3, pandas, numpy, matplotlib, seaborn, shap (opcional).
- **Artefactos guardados** en `preprocesados/`:
  - `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`
  - `pipeline.pkl` (ColumnTransformer ajustado)
  - `label_encoder.pkl`
  - `modelo_final.pkl` (HistGradientBoosting optimizado)
  - `modelo_final_meta.pkl` (nombre, params, scores)

---

*Documento generado a partir de la ejecución completa del notebook `EDA_Preprocesamiento_Sleep_Health.ipynb`.*
