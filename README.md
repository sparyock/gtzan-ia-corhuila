# 🎵 GTZAN Music Genre Classification - CNN (Corhuila 2026-2)

Proyecto Final para la asignatura de **Inteligencia Artificial** - *Corporación Universitaria del Huila (CORHUILA)*.

## 👥 Integrantes del Equipo
* **Yeison Andrés Scarpeta Díaz** (Líder de IA / Modelado y Entrenamiento)
* **Nicolas Gabriel Alvarez Aguirre** (Gestión de Datos, Métricas e Informe APA 7)
* **Sergio Luis Angel Romero** (Desarrollo Web en Flask y Despliegue)

**Docente:** Ing. José Miguel Llanos Mosquera  
**Programa:** Ingeniería de Sistemas  
**Fecha de Sustentación:** Martes, 10 de Noviembre de 2026 (6:00 AM - C5-508)  

---

## 📌 Descripción del Proyecto
El objetivo principal de este proyecto es clasificar automáticamente **10 géneros musicales** a partir de archivos de audio, abordándolo desde una perspectiva de **Visión por Computadora**. 

Los archivos de audio del dataset GTZAN se transforman en **Espectrogramas Mel** (imágenes 2D de frecuencia/tiempo) que son procesados mediante una **Red Neuronal Convolucional (CNN)** implementada en TensorFlow/Keras.

### 🎷 Géneros a Clasificar (10 Clases)
`Blues`, `Classical`, `Country`, `Disco`, `Hiphop`, `Jazz`, `Metal`, `Pop`, `Reggae`, `Rock`.

---

## 🏗️ Arquitectura de la Red Neuronal (CNN)
La red está optimizada para evitar el sobreajuste (*overfitting*) y procesa espectrogramas de tamaño **128x128 píxeles en escala de grises**:

1. **Capa de Normalización (`Rescaling`):** Escala los valores de los píxeles de `[0, 255]` a `[0.0, 1.0]`.
2. **Bloques Convolucionales:**
   * **Bloque 1:** Conv2D (32 filtros, 3x3, ReLU) + MaxPooling2D (2x2)
   * **Bloque 2:** Conv2D (64 filtros, 3x3, ReLU) + MaxPooling2D (2x2)
   * **Bloque 3:** Conv2D (128 filtros, 3x3, ReLU) + MaxPooling2D (2x2) + Dropout (0.3)
3. **Clasificación Densa:**
   * Flatten
   * Dense (64 neuronas, ReLU) + Dropout (0.4)
   * Dense (10 neuronas, Softmax)

---

## 📊 Resultados y Evaluación Técnica
El modelo alcanzó un rendimiento muy estable en el conjunto de validación sin sobreajuste:
* **Accuracy de Validación:** `~62%`
* **Loss de Validación:** `~1.18`

Las evidencias gráficas requeridas se generan automáticamente en `web_app/static/`:
* `training_plot.png`: Curvas de aprendizaje (Precisión y Pérdida).
* `confusion_matrix.png`: Matriz de Confusión por género musical.
* `classification_report.txt`: Reporte métrico detallado (Precision, Recall, F1-Score).

---

## 📂 Estructura del Repositorio

```text
gtzan-ia-corhuila/
│
├── dataset/
│   ├── images_original/   # Espectrogramas base de Kaggle
│   └── procesado/         # Partición automática (Train 80% / Val 20%)
│
├── web_app/
│   ├── model/
│   │   ├── modelo_final.keras  # Modelo CNN entrenado
│   │   └── clases.json         # Mapeo de índices a géneros musicales
│   └── static/                 # Gráficas de evaluación técnica (PNG)
│
├── env/                   # Entorno virtual de Python
├── split_data.py          # Script de separación de datos (evita Data Leakage)
├── train_model.py         # Script de arquitectura, entrenamiento y evaluación
└── README.md              # Documentación oficial del repositorio
