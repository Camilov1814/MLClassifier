# IA - Inteligencia Artificial

Repositorio de la materia de Inteligencia Artificial. Proyectos de Machine Learning siguiendo la metodología CRISP-DM.

## Estructura del Repositorio

```
IA/
├── EDA_Preprocesamiento_Social_Media.ipynb   # EDA + Preprocesamiento (Fases 2 y 3 CRISP-DM)
├── Social_media_impact_on_life.csv           # Dataset principal
├── airline_agrupamiento.ipynb                # Proyecto de agrupamiento (clustering)
├── .gitignore
└── README.md
```

## Proyecto: Impacto de Redes Sociales en Estudiantes

**Objetivo:** Clasificar el impacto general (`Overall_Impact`: Negative, Neutral, Positive) del uso de redes sociales en estudiantes.

**Dataset:** 1,705 registros, 11 variables (demográficas, uso de redes, salud mental, rendimiento académico).

### Fases completadas

- **Fase 2 - Comprensión de Datos (EDA):** Análisis exploratorio completo con distribuciones, correlaciones, análisis bivariado y detección de outliers.
- **Fase 3 - Preparación de Datos:** Pipeline de preprocesamiento con `ColumnTransformer` (StandardScaler/MinMaxScaler para numéricas, OrdinalEncoder para ordinales, OneHotEncoder para nominales). Comparación de métodos de escalado.

### Stack

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn (Pipelines, ColumnTransformer, clasificadores)

## Autor

Camilo - Ingeniería de Sistemas
