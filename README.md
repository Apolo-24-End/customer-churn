# Customer Churn Prediction

Sistema de predicción de abandono de clientes construido sobre un dataset de 1 millón de registros. Incluye un pipeline completo de Machine Learning y una API con dashboard interactivo.

**Demo en vivo:** [customer-churn-elop.onrender.com](https://customer-churn-elop.onrender.com)

> El servidor en Render es de tier gratuito — si lleva un rato inactivo, la primera carga puede tardar ~30-60 segundos.

---

## Stack

| Capa | Tecnologías |
|---|---|
| Datos y features | pandas, numpy, scikit-learn, imbalanced-learn (SMOTE) |
| Modelos | LightGBM, XGBoost, Stacking Ensemble, Optuna |
| API | FastAPI, uvicorn |
| Frontend | HTML / CSS / JS, Chart.js |

---

## Estructura del proyecto

```
├── src/
│   ├── data/          # Carga y preprocesamiento
│   ├── eda/           # Análisis exploratorio
│   ├── features/      # Ingeniería de variables y selección
│   ├── models/        # Entrenamiento, tuning y evaluación
│   └── pipeline.py    # Orquestador del flujo completo
├── api/
│   ├── routers/       # Endpoints: /eda, /model, /predict
│   └── main.py        # App FastAPI
├── frontend/          # Dashboard estático servido por la API
├── models/            # Artefactos entrenados (.joblib)
├── outputs/           # Resultados del pipeline (.json)
├── config.py          # Configuración global
└── run.py             # Punto de entrada principal
```

---

## Cómo ejecutarlo localmente

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Añadir el dataset

Coloca el archivo `customer_churn_1M.csv` en la carpeta `data/` (no está en el repositorio por su tamaño de 162 MB).

### 3. Ejecutar

```bash
# Solo el pipeline de entrenamiento
python run.py pipeline

# Solo la API (requiere haber ejecutado el pipeline antes)
python run.py api

# Pipeline completo + API
python run.py all
```

La API queda disponible en `http://localhost:8000`.

---

## Notas de despliegue

- El CSV está excluido del repositorio. Los modelos y outputs ya están pre-computados y commiteados.
- Render auto-despliega desde GitHub en cada push a `main`.
- Comando de inicio en Render: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
