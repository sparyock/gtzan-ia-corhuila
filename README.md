# Clasificación de Géneros Musicales con CNN (GTZAN) - Proyecto IA

Este repositorio contiene el proyecto final para la asignatura de Inteligencia Artificial de la Corporación Universitaria del Huila (CORHUILA). Consiste en un sistema basado en Redes Neuronales Convolucionales (CNN) capaz de clasificar audios en 9 géneros musicales diferentes a partir de espectrogramas Mel de 3 segundos.

## 👥 Equipo de Trabajo
* Yeison Andrés Scarpeta Díaz  - *Entrenamiento del modelo IA y procesamiento de datos*
* Angel - *Desarrollo e integración de la Web App (Flask)*
* Nicolás - *Documentación, reportes APA y análisis de resultados*

## 📊 Dataset y Clases
Se utilizó el dataset **"3-Second Melspectrograms for GTZAN"** extraído de Kaggle, el cual contiene 9,000 imágenes de espectrogramas generadas con Librosa. 

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

*(Nota técnica: El género "Jazz" fue excluido en esta versión del modelo según la disponibilidad del dataset).*

## 🧠 Arquitectura y Rendimiento del Modelo
El entrenamiento pesado se migró de CPU local a **Google Colab (Aceleración por GPU T4)** para procesar las 9,000 imágenes de manera eficiente.

* **Arquitectura:** Red Neuronal Convolucional (CNN) con 4 bloques convolucionales, capas de MaxPooling, Dropout para evitar sobreajuste y Data Augmentation integrado.
* **Épocas de entrenamiento:** **35 épocas**.
* **Learning Rate:** **0.001** (optimizador Adam).
* **Batch Size:** 32.
* **Entrada:** espectrogramas Mel de 128x128 px en escala de grises.
* **Salida:** 9 géneros musicales (softmax).
* **Artefacto final:** `modelo_final.keras` (Guardado en `web_app/model/`).

## ✅ Resultados de la Última Evaluación
Evaluación reproducible del modelo guardado sobre el conjunto de validación (900 imágenes, 100 por género):

| Métrica | Valor |
| --- | --- |
| Accuracy | **75.00%** |
| Loss | **0.7499** |
| Precision (macro) | 0.761 |
| Recall (macro) | 0.750 |
| F1-score (macro) | 0.749 |
| Épocas | 35 |
| Learning Rate | 0.001 |
| Umbral exigido | 75% |
| ¿Cumple? | **Sí** |

Los resultados completos se guardan automáticamente en:

* `web_app/static/evaluation_results.json` (métricas, matriz de confusión y detalle por clase).
* `web_app/static/classification_report.txt` (reporte de clasificación).
* `web_app/static/training_plot.png` (curvas de Accuracy y Loss).
* `web_app/static/confusion_matrix.png` (matriz de confusión).
* `web_app/static/per_class_metrics.png` (precision/recall/F1 por género).

Para reproducir la evaluación:

```bash
python evaluate_model.py
```

La aplicación web muestra todo en la ruta **`/resultados`**.

> Nota: si una nueva evaluación queda por debajo del 75%, el script lo indica y se debe
> ajustar `LEARNING_RATE = 0.001` en `train_model.py` y reentrenar.

## 🚀 Ejecución Local
```bash
# 1. Instalar dependencias
pip install -r web_app/requirements.txt

# 2. Entrenar (opcional, pesado en CPU)
python train_model.py

# 3. Evaluar el modelo ya entrenado
python evaluate_model.py

# 4. Levantar la aplicación web
python web_app/app.py
```

Abrir `http://localhost:5000`. Rutas disponibles:

* `/` → clasificador de audio `.wav`.
* `/resultados` → métricas y gráficas del modelo.
* `/health` → estado del servicio en formato JSON.

## ☁️ Despliegue en Render
El archivo `Procfile` de la raíz define el comando de producción:

```text
web: gunicorn web_app.app:app --bind 0.0.0.0:$PORT --timeout 180 --workers 1
```

Pasos: subir el repositorio a GitHub, crear un **Web Service** en Render, conectar el repo,
usar `pip install -r web_app/requirements.txt` como build command y el `Procfile` como start command.

## 📁 Estructura del Proyecto
```text
gtzan-ia-corhuila/
│
├── dataset/                 # (Ignorado en Git) 9,000 Espectrogramas de entrenamiento y validación.
├── web_app/                 # Directorio principal de la aplicación web.
│   ├── model/               # Contiene el modelo IA (modelo_final.keras) y clases.json.
│   ├── static/              # Gráficas de rendimiento (matriz de confusión, curvas) y reporte txt.
│   ├── templates/           # Vistas HTML de la aplicación web (index.html y resultados.html).
│   └── app.py               # Script principal de Flask.
│
├── train_model.py           # Script principal de entrenamiento de la CNN.
├── evaluate_model.py        # Evaluación reproducible del modelo y generación de métricas.
├── split_data.py            # Script auxiliar para partición de datos.
├── Procfile                 # Comando de arranque para Render/Gunicorn.
├── .gitignore               # Exclusión de archivos pesados y entornos virtuales.
└── README.md                # Documentación del proyecto.
