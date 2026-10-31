"""
==============================================================================
PROYECTO FINAL INTELIGENCIA ARTIFICIAL - CORHUILA (2026-2)
Módulo: Entrenamiento y Evaluación de Red Neuronal Convolucional (CNN)
Autor / Responsable IA: Yeison Andrés Scarpeta Díaz
==============================================================================
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ------------------------------------------------------------------------------
# 1. CONFIGURACIÓN GLOBAL Y RUTAS MULTIPLATAFORMA
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_TRAIN = BASE_DIR / "dataset" / "procesado" / "train"
DATASET_VAL = BASE_DIR / "dataset" / "procesado" / "val"
MODEL_SAVE_PATH = BASE_DIR / "web_app" / "model" / "modelo_final.keras"
CLASSES_JSON_PATH = BASE_DIR / "web_app" / "model" / "clases.json"
STATIC_DIR = BASE_DIR / "web_app" / "static"

BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 35                  # Aumentado a 35 épocas según requerimiento
LEARNING_RATE = 0.001        # Ajustado a 0.001 según requerimiento

CLASES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "metal",
    "pop",
    "reggae",
    "rock",
]


# ------------------------------------------------------------------------------
# 2. CARGA Y PREPROCESAMIENTO DE DATOS
# ------------------------------------------------------------------------------
def cargar_datasets():
    print("-> Cargando conjunto de datos de entrenamiento y validación...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_TRAIN,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True,
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_VAL,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
    )

    # Exportar archivo JSON con mapeo de clases para Flask
    CLASSES_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping = {i: clase for i, clase in enumerate(CLASES)}
    with open(CLASSES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=4, ensure_ascii=False)
    print(f"-> Mapeo de clases exportado en: {CLASSES_JSON_PATH}")

    # Optimización de carga en memoria
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds


# ------------------------------------------------------------------------------
# 3. ARQUITECTURA CNN ROBUSTA Y COMPATIBLE
# ------------------------------------------------------------------------------
def construir_cnn():
    print("-> Diseñando arquitectura CNN con Data Augmentation...")

    # Capa de aumento de datos para multiplicar el volumen visualmente
    data_augmentation = tf.keras.Sequential([
        layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
        layers.RandomContrast(factor=0.15)
    ])

    model = models.Sequential(
        [
            layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
            data_augmentation,
            layers.Rescaling(1.0 / 255.0),  # Normalización de píxeles
            
            # Bloque Convolucional 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            
            # Bloque Convolucional 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            
            # Bloque Convolucional 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Bloque Convolucional 4 (Extra para mayor volumen de datos)
            layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Capas Densas y Clasificación
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.4),
            layers.Dense(len(CLASES), activation="softmax"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ------------------------------------------------------------------------------
# 4. EVALUACIÓN Y GENERACIÓN DE EVIDENCIAS GRÁFICAS
# ------------------------------------------------------------------------------
def generar_reporte_grafico(history, model, val_ds):
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    print("-> Generando gráficas de evaluación técnica...")

    # 1. Curvas de Aprendizaje (Accuracy y Loss)
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs_range, acc, "b-o", label="Precisión Entrenamiento")
    ax1.plot(epochs_range, val_acc, "r-s", label="Precisión Validación")
    ax1.set_title("Evolución de la Precisión (Accuracy)")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Precisión")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower right")

    ax2.plot(epochs_range, loss, "b-o", label="Pérdida Entrenamiento")
    ax2.plot(epochs_range, val_loss, "r-s", label="Pérdida Validación")
    ax2.set_title("Evolución de la Pérdida (Loss)")
    ax2.set_xlabel("Épocas")
    ax2.set_ylabel("Pérdida")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plot_path = STATIC_DIR / "training_plot.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"   [+] Curvas guardadas en: {plot_path}")

    # 2. Matriz de Confusión
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASES,
        yticklabels=CLASES,
    )
    plt.title("Matriz de Confusión - Evaluación del Modelo")
    plt.ylabel("Género Real")
    plt.xlabel("Género Predicho")
    plt.tight_layout()
    cm_path = STATIC_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"   [+] Matriz de confusión guardada en: {cm_path}")

    # 3. Reporte Métrico de Clasificación
    report = classification_report(
        y_true, y_pred, target_names=CLASES, zero_division=0
    )
    report_path = STATIC_DIR / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"   [+] Reporte métrico detallado guardado en: {report_path}")


# ------------------------------------------------------------------------------
# 5. EJECUCIÓN PRINCIPAL
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    train_ds, val_ds = cargar_datasets()
    model = construir_cnn()
    model.summary()

    cb_checkpoint = callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    )

    print("\n=======================================================")
    print("  INICIANDO ENTRENAMIENTO ROBUSTO DE LA RED NEURONAL")
    print("=======================================================\n")

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[cb_checkpoint]
    )

    print("\n=======================================================")
    print("  ENTRENAMIENTO COMPLETADO - EVALUANDO MODELO")
    print("=======================================================\n")

    best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    generar_reporte_grafico(history, best_model, val_ds)

    print("\n¡Proceso finalizado con éxito! Entregables listos:")
    print(f"1. Modelo Keras: {MODEL_SAVE_PATH}")
    print(f"2. Mapeo JSON: {CLASSES_JSON_PATH}")
    print(f"3. Gráficas estáticas en: {STATIC_DIR}")