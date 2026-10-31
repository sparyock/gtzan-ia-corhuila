# Clasificación de Géneros Musicales con CNN (GTZAN) - Proyecto IA

Este repositorio contiene el proyecto final para la asignatura de Inteligencia Artificial de la Corporación Universitaria del Huila (CORHUILA). Consiste en un sistema basado en Redes Neuronales Convolucionales (CNN) capaz de clasificar audios en 9 géneros musicales diferentes a partir de espectrogramas Mel de 3 segundos.

## 👥 Equipo de Trabajo
* Yeison Andrés Scarpeta Díaz (Jason) - *Entrenamiento del modelo IA y procesamiento de datos*
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
* **Épocas de entrenamiento:** 35 épocas.
* **Precisión final (Accuracy):** **75.00%** (Superando el umbral exigido del 75%).
* **Artefacto final:** `modelo_final.keras` (Guardado en `web_app/model/`).

## 📁 Estructura del Proyecto
```text
gtzan-ia-corhuila/
│
├── dataset/                 # (Ignorado en Git) 9,000 Espectrogramas de entrenamiento y validación.
├── web_app/                 # Directorio principal de la aplicación web.
│   ├── model/               # Contiene el modelo IA (modelo_final.keras) y clases.json.
│   ├── static/              # Gráficas de rendimiento (matriz de confusión, curvas) y reporte txt.
│   ├── templates/           # Vistas HTML de la aplicación web.
│   └── app.py               # Script principal de Flask.
│
├── train_model.py           # Script principal de entrenamiento de la CNN.
├── split_data.py            # Script auxiliar para partición de datos.
├── .gitignore               # Exclusión de archivos pesados y entornos virtuales.
└── README.md                # Documentación del proyecto.
