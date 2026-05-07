# Clasificación de Riesgo de Trastorno del Sueño

Proyecto de Machine Learning siguiendo metodología **CRISP-DM** para predecir el nivel de riesgo de trastorno del sueño a partir de variables fisiológicas, conductuales y demográficas.

## Problema

Clasificación multiclase del riesgo de trastorno del sueño en 4 niveles (`Healthy`, `Mild`, `Moderate`, `Severe`) usando 100,000 registros con 32 variables.

## Estructura del repositorio

```
IA/
├── Sleep_Health_Clasificacion.ipynb   # Notebook principal (CRISP-DM completo)
├── DOCUMENTACION_Sleep_Health.md      # Documentación explicativa de decisiones
├── sleep_health_dataset.csv           # Dataset (100k filas)
├── sleep_Health_Orange.ows            # Workflow de Orange Data Mining
├── preprocesados/                     # Artefactos generados (gitignored)
├── README.md
└── .gitignore
```

## Resultados

| Modelo | Recall (test) | F1-weighted (test) |
|---|---:|---:|
| **HistGradientBoosting** (final) | **0.954** | **0.953** |
| Random Forest (challenger) | 0.891 | 0.885 |

- Modelo final: `HistGradientBoosting` con `learning_rate=0.1, max_iter=200, max_depth=None`.
- Hallazgo clave: con solo **15 features** (vs 63 originales) se mantiene ~98% del rendimiento.
- Sin evidencia de data leakage estructural (verificado con experimento de ablación).

## Estructura del notebook

1. Imports y carga
2. EDA (estructura, target, correlaciones)
3. Preparación de datos (ColumnTransformer)
4. Modelado (GridSearchCV + 10-Fold CV con 8 clasificadores)
5. Evaluación HGB vs Random Forest
6. Selección de características (7 métodos: Correlación, ANOVA, RFECV, RF importance, Permutation, SHAP, Mutual Information)
7. Experimentación con features (4 experimentos: top 3, peor 10, top-15 SelectKBest, sin sospechosas)
8. Conclusiones

## Cómo ejecutar

```bash
# Instalar dependencias
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib jupyter

# Abrir notebook
jupyter notebook Sleep_Health_Clasificacion.ipynb
```

El notebook tarda ~10-15 minutos en ejecutarse completo. La sección 4 (GridSearchCV) es la más pesada; las secciones 6-7 reutilizan los modelos optimizados guardados en `preprocesados/best_estimators.pkl` para no re-tunear.

## Stack

- **Python 3.10+**
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` (pipelines, GridSearchCV, ensembles)
- `shap` (explicabilidad)
- `joblib` (serialización)

## Documentación

Para entender el **razonamiento detrás de cada decisión** (preprocesamiento, métrica, modelo, feature selection, experimentos), ver [DOCUMENTACION_Sleep_Health.md](./DOCUMENTACION_Sleep_Health.md).

## Autor

Camilo — Ingeniería de Sistemas
