# Clasificación de Géneros Musicales con CNN (GTZAN) - Proyecto IA

Este repositorio contiene el proyecto final para la asignatura de **Inteligencia Artificial** de la Corporación Universitaria del Huila (CORHUILA). Consiste en un sistema basado en Redes Neuronales Convolucionales (CNN) capaz de clasificar audios en **9 géneros musicales diferentes** a partir de espectrogramas Mel de 3 segundos.

## 👥 Equipo de Trabajo

- **Yeison Andrés Scarpeta Díaz** – *Entrenamiento del modelo IA y procesamiento de datos.*
- **Angel** – *Desarrollo e integración de la Web App (Flask).*
- **Nicolás** – *Documentación, reportes APA y análisis de resultados.*

## 📊 Dataset y Clases

Se utilizó el dataset **3-Second Melspectrograms for GTZAN**, extraído de Kaggle, del cual se procesaron **8.100 imágenes útiles**, distribuidas de la siguiente manera:

- **7.200 imágenes** para entrenamiento.
- **900 imágenes** para validación.

El modelo fue entrenado para clasificar los siguientes **9 géneros musicales**:

1. Blues
2. Classical
3. Country
4. Disco
5. Hiphop
6. Metal
7. Pop
8. Reggae
9. Rock

> **Nota técnica:** El género Jazz fue excluido en esta versión del modelo según la disponibilidad del dataset.

## 🧠 Arquitectura y Rendimiento del Modelo

El entrenamiento se migró de CPU local a **Google Colab con aceleración por GPU T4**, permitiendo procesar las 8.100 imágenes de manera eficiente.

- **Arquitectura:** Red Neuronal Convolucional (CNN) con 4 bloques convolucionales, capas MaxPooling, Dropout y Data Augmentation.
- **Épocas de entrenamiento:** 35.
- **Learning Rate:** 0.001.
- **Optimizador:** Adam.
- **Batch Size:** 32.
- **Entrada:** Espectrogramas Mel de 128 × 128 píxeles en escala de grises.
- **Salida:** 9 clases mediante función Softmax.
- **Modelo final:** `modelo_final.keras`.
- **Ubicación:** `web_app/model/`.

## ✅ Resultados de la Última Evaluación

Evaluación reproducible del modelo guardado sobre el conjunto de validación de 900 imágenes (100 por género).

| Métrica | Resultado |
|---|---:|
| Accuracy | **75.00 %** |
| Loss | **0.7499** |
| Precision (macro) | 0.761 |
| Recall (macro) | 0.750 |
| F1-score (macro) | 0.749 |
| Épocas | 35 |
| Learning Rate | 0.001 |
| Umbral exigido | 75 % |
| Cumplimiento | **Sí** |

Los resultados completos se guardan automáticamente en:

- `web_app/static/evaluation_results.json` – Métricas, matriz de confusión y detalle por clase.
- `web_app/static/classification_report.txt` – Reporte de clasificación.
- `web_app/static/training_plot.png` – Curvas de Accuracy y Loss.
- `web_app/static/confusion_matrix.png` – Matriz de confusión.
- `web_app/static/per_class_metrics.png` – Precision, Recall y F1 por género.

**Para reproducir la evaluación:**

```bash
python evaluate_model.py
```

La aplicación web permite consultar todos los resultados desde la ruta `/resultados`.

> **Nota:** Si una nueva evaluación queda por debajo del 75 %, se debe revisar el parámetro `LEARNING_RATE = 0.001` en `train_model.py` y realizar un nuevo entrenamiento.

## 🚀 Ejecución Local

### 1. Instalar dependencias

```bash
pip install -r web_app/requirements.txt
```

### 2. Entrenar el modelo (opcional)

```bash
python train_model.py
```

> El entrenamiento puede requerir recursos computacionales elevados si se ejecuta en CPU.

### 3. Evaluar el modelo entrenado

```bash
python evaluate_model.py
```

### 4. Ejecutar la aplicación web

```bash
python web_app/app.py
```

Abrir en el navegador:

`http://localhost:5000`

### Rutas disponibles

| Ruta | Funcionalidad |
|---|---|
| `/` | Clasificador de archivos de audio `.wav`. |
| `/resultados` | Visualización de métricas y gráficas del modelo. |
| `/health` | Estado del servicio en formato JSON. |




## 📁 Estructura del Proyecto

```text
gtzan-ia-corhuila/
│
├── dataset/                 # 8.100 espectrogramas (ignorado en Git).
│
├── web_app/                 # Aplicación web principal.
│   ├── model/               # Modelo IA y archivo clases.json.
│   ├── static/              # Gráficas, métricas y reportes.
│   ├── templates/            # Plantillas HTML de la aplicación.
│   └── app.py               # Aplicación principal Flask.
│
├── train_model.py           # Entrenamiento de la CNN.
├── evaluate_model.py        # Evaluación y generación de métricas.
├── split_data.py            # Partición del conjunto de datos.
├── Procfile                 # Configuración de Render/Gunicorn.
├── .gitignore               # Archivos excluidos del repositorio.
└── README.md                # Documentación del proyecto.
```

---

**Proyecto académico desarrollado para la asignatura de Inteligencia Artificial.**

**Corporación Universitaria del Huila (CORHUILA).**
