# Documentación — Clasificación de Riesgo de Trastorno del Sueño

> Documento explicativo del notebook `Sleep_Health_Clasificacion.ipynb`. El foco está en **por qué** se tomó cada decisión, no en repetir los outputs (que están en el notebook).

---

## 1. El problema y los datos

Predecir el nivel de riesgo de trastorno del sueño (`sleep_disorder_risk` con 4 niveles: Healthy / Mild / Moderate / Severe) a partir de variables fisiológicas, conductuales y demográficas.

Dataset de 100,000 registros con 32 columnas, sin nulos ni duplicados. El target está moderadamente desbalanceado: ~54% Healthy y solo ~4% Severe. Esto define la métrica y el manejo del desbalance.

---

## 2. Decisiones de preprocesamiento y por qué

### Codificación del target
Se usa `LabelEncoder` con orden ordinal (`Healthy < Mild < Moderate < Severe`). El target *tiene* orden natural (riesgo creciente), así que codificarlo numéricamente respetando ese orden permite además calcular correlaciones interpretables con variables continuas.

### Tres tipos de variables, tres tratamientos
- **23 variables numéricas con `StandardScaler`**: necesario para modelos basados en distancia (KNN) o gradiente (LogReg, SVM lineal). Los modelos de árboles no lo necesitan, pero estandarizar no les afecta — usar un solo pipeline simplifica el flujo.
- **`chronotype` con `OrdinalEncoder`**: tiene orden natural (`Morning < Neutral < Evening`). One-Hot lo trataría como tres categorías independientes y perdería esa información.
- **6 variables nominales con `OneHotEncoder(drop='if_binary')`**: el resto no tiene orden. `drop='if_binary'` evita redundancia en variables como `day_type` (con dos categorías genera una sola columna).

### Variables con muchas categorías (`country` con 15, `occupation` con 12)
Se tratan igual que el resto de nominales. Son cardinalidades manejables y todas las categorías tienen suficientes muestras. La feature selection confirmó después que individualmente aportan poco — pero el modelo lo aprende solo, no se necesitan estrategias especiales (target encoding, agrupamiento, etc.) que serían riesgosas o injustificadas para este volumen.

### Partición 80/20 estratificada
20,000 instancias en test es suficiente para evaluación robusta. La estratificación es **obligatoria** con clases minoritarias: garantiza que `Severe` aparezca en proporciones iguales en train y test.

### El `ColumnTransformer` se ajusta solo en train
Se usa `fit_transform(X_train)` y luego `transform(X_test)`. Esto evita data leakage de preprocesamiento: el `StandardScaler` calcula media/desviación solo con train, y el `OneHotEncoder` fija las categorías solo con lo visto en train.

---

## 3. Decisiones de modelado

### ¿Por qué `recall_weighted` como métrica?
En clasificación clínica, **detectar correctamente los casos en riesgo importa más que evitar falsas alarmas**. Recall mide la sensibilidad por clase; el ponderado se ajusta al desbalance del target. Un falso negativo en clase `Severe` es mucho más costoso que un falso positivo.

### ¿Por qué muestra estratificada de 20k para tuning?
Con 100k filas, el GridSearchCV completo sería innecesariamente lento. Una muestra estratificada de 20k preserva las proporciones de clase y produce hiperparámetros prácticamente idénticos a los que daría el dataset completo. La diferencia se valida empíricamente en la sección 5 (gap CV-Test bajo).

### ¿Por qué GridSearchCV exhaustivo y no RandomizedSearch?
Con grids compactos (3-6 valores por hiperparámetro), GridSearch cubre todas las combinaciones en pocos minutos. Es **determinístico, reproducible y más fácil de explicar** en presentación. RandomizedSearch tiene sentido cuando el espacio es enorme (cientos de combos), no es el caso aquí.

### ¿Por qué tunear primero y comparar después?
La comparación justa es entre las **mejores versiones** de cada modelo, no entre defaults. Un modelo "malo con defaults" puede ser excelente con tuning. Por eso primero optimizamos cada uno por separado y luego los enfrentamos en CV final.

### Reemplazos de modelos por velocidad
- `SVC(rbf)` → `LinearSVC`: el SVM con kernel RBF es O(n²); inviable en 80k filas. LinearSVC mantiene la idea de máquina de soporte vectorial pero solo con frontera lineal, lo que basta porque el ranking final mostró que los modelos lineales topan en ~81% — RBF habría quedado como challenger sin amenazar al ganador.
- `GradientBoostingClassifier` → `HistGradientBoostingClassifier`: 10× más rápido gracias a binning por histogramas, mismo poder predictivo. Reemplazo sin pérdida.

---

## 4. Por qué dos finalistas: HGB vs RF

Aunque HGB gana en métricas, conservar Random Forest como challenger tiene tres razones:

1. **Robustez del análisis**: si los hallazgos coinciden en ambos (mismas features importantes, misma reacción a experimentos), las conclusiones son más sólidas e independientes del modelo.
2. **Trade-off rendimiento vs interpretabilidad**: RF expone `feature_importances_` de forma más natural ("cuántos splits usaron esta variable"). Para presentar a un cliente o stakeholder no técnico, RF es conceptualmente más amigable. La diferencia de 5-7 puntos en F1 puede no ser crítica si la interpretabilidad pesa más.
3. **Diferencia mecánica útil**: RF promedia árboles independientes (reduce varianza); HGB suma árboles que corrigen errores anteriores (reduce sesgo). Cuando RF y HGB rinden parecido los datos son aditivos; cuando HGB gana por mucho, hay interacciones complejas que solo el boosting captura — ese diagnóstico es valioso.

**Recomendación final del proyecto**: HGB para producción, RF como referencia interpretable.

---

## 5. Feature Selection: qué se hizo y por qué

Se combinaron **7 métodos complementarios** en lugar de uno solo, porque cada uno tiene sesgos distintos. La idea es que una feature genuinamente importante aparezca consistentemente entre los rankings de varios métodos.

| Método | Tipo | Captura | Limitación |
|---|---|---|---|
| Correlación con target | Univariado lineal | Asociación lineal directa | Ignora interacciones y no linealidad |
| ANOVA F-test | Univariado lineal | Diferencia significativa entre clases | Asume relaciones lineales |
| RFECV | Wrapper | Importancia evaluando subsets reales del modelo | Costoso, depende del estimador base |
| Importancia de RF | Embedded | Uso real en splits de árboles | Sesgada a features con muchos valores únicos |
| Permutation Importance (HGB) | Modelo-agnóstico | Pérdida real al permutar la feature | Computacionalmente costosa |
| SHAP | Explicabilidad | Contribución marginal por instancia (teoría de juegos) | Costo computacional alto |
| Mutual Information | Univariado no lineal | Dependencia general, no solo lineal | Ignora interacciones |

Una feature aparece en el "consenso" si está en el top-15 de **4+ métodos** (sobre 7). Este umbral filtra ruido específico de cada método y refuerza features genuinamente importantes.

**El propósito es identificar redundancia e irrelevancia, no detectar leakage.** El leakage requiere análisis del proceso de generación de datos, no estadísticas — se aborda con el experimento D más adelante.

---

## 6. Los 4 experimentos: qué se está probando y por qué

| Exp | Pregunta de negocio | Hipótesis a verificar |
|---|---|---|
| **A** | ¿Llegamos lejos con poco? | Con solo 3 features bien elegidas se logra una fracción notable del rendimiento |
| **B** | ¿La feature selection funciona? | Las features mal calificadas son realmente ruido |
| **C** | ¿El plateau de SelectKBest es real? | Top 15 features ≈ baseline completo |
| **D** | ¿Hay data leakage? | Quitar las 2 más correlacionadas no hunde el modelo (sería redundancia, no leakage) |

### Razonamiento de las features elegidas en A
Tres dimensiones distintas para validar que la información está distribuida:
- `mental_health_condition_Healthy`: estado mental (la categoría con mayor peso individual del análisis de robustez previo)
- `cognitive_performance_score`: función cognitiva (mayor correlación con el target)
- `stress_score`: factor conductual/emocional

Si las tres juntas alcanzan rendimiento decente, el modelo no depende de variables específicas sino de la **combinación de dimensiones**.

### Por qué el experimento D no es prueba definitiva de no-leakage
Si quitamos las features sospechosas y el modelo se mantiene, **descartamos** que dependa críticamente de ellas y reforzamos que la información estaba redundante. Pero el leakage estructural real solo se descarta entendiendo cómo se generó el dataset (¿`sleep_disorder_risk` se calculó usando `sleep_quality_score`?). Como el dataset parece sintético sin documentación pública del proceso generador, este experimento es lo más cercano a un check empírico.

---

## 7. Hallazgos principales

### Sobre el modelo
- HGB supera consistentemente a RF, especialmente en clases minoritarias (Moderate, Severe). Esto es valioso clínicamente: detectar mejor a quienes están en riesgo es el objetivo central.
- El gap CV-Test es bajo en ambos modelos → **no hay overfitting**.
- AdaBoost colapsa en la clase Severe (F1≈0). No es buen candidato para datasets desbalanceados con clases minoritarias raras.

### Sobre las features
- Las dimensiones predictivas reales son: estado de salud mental, rendimiento cognitivo, calidad/duración del sueño, estrés. Variables demográficas (`country`, `occupation`) aportan poco individualmente.
- La información está **redundante**: con 15 features bien elegidas se logra ~98% del rendimiento del modelo con 63. Esto permite construir un modelo de producción más liviano.

### Sobre data leakage
- Quitar las dos features más correlacionadas con el target solo causa una caída marginal. La información estaba **redundante con otras features**, no leakeada estructuralmente.
- Confirmación adicional: HGB y RF reaccionan de la misma forma a los experimentos → el comportamiento no es artefacto de un modelo específico.

---

## 8. Limitaciones

- **Dataset posiblemente sintético**: distribuciones muy limpias, sin nulos, sin valores raros. Generalización a datos clínicos reales requiere validación externa.
- **No se exploraron técnicas avanzadas de balanceo** (SMOTE, undersampling, focal loss). Solo `class_weight='balanced'` en los modelos compatibles.
- **Sin calibración de probabilidades**: si el modelo se usa como tamizaje médico, conviene aplicar calibración isotónica para que las probabilidades sean interpretables.
- **No se hizo análisis de subgrupos**: equidad entre países, géneros, ocupaciones. Importante si el modelo se despliega.

---

## 9. Reproducibilidad

- `random_state=42` en todos los splits y modelos.
- Artefactos guardados en `preprocesados/`:
  - `pipeline.pkl` — `ColumnTransformer` ajustado con train.
  - `label_encoder.pkl` — encoder ordinal del target.
  - `best_estimators.pkl` — diccionario con los 8 modelos optimizados (resultado del GridSearch).
  - `gs_df.pkl` — tabla de resultados del GridSearch.
  - `modelo_final.pkl` — HGB re-entrenado con train completo.
- El guardado de `best_estimators` y `gs_df` permite **rehacer las secciones 6-7 sin tener que volver a ejecutar el GridSearch** (que tarda decenas de minutos).
- Para predecir nuevos datos: cargar `pipeline` + `modelo_final` + `label_encoder` → `pipeline.transform(X_new)` → `modelo.predict(...)` → `label_encoder.inverse_transform(...)`.
