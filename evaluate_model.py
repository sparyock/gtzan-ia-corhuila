"""
==============================================================================
PROYECTO FINAL INTELIGENCIA ARTIFICIAL - CORHUILA (2026-2)
Modulo: Evaluacion del modelo entrenado sobre el conjunto de validacion
==============================================================================
Uso:
    python evaluate_model.py

Genera:
    web_app/static/evaluation_results.json
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATASET_VAL = BASE_DIR / "dataset" / "procesado" / "val"
MODEL_PATH = BASE_DIR / "web_app" / "model" / "modelo_final.keras"
CLASSES_JSON_PATH = BASE_DIR / "web_app" / "model" / "clases.json"
STATIC_DIR = BASE_DIR / "web_app" / "static"
RESULTS_JSON_PATH = STATIC_DIR / "evaluation_results.json"
PER_CLASS_PLOT_PATH = STATIC_DIR / "per_class_metrics.png"

BATCH_SIZE = 32
IMG_SIZE = (128, 128)
UMBRAL = 0.75


def cargar_clases():
    with open(CLASSES_JSON_PATH, "r", encoding="utf-8") as f:
        clases_dict = json.load(f)
    return [clases_dict[str(i)] for i in range(len(clases_dict))]


def main():
    CLASES = cargar_clases()
    print(f"-> Clases: {CLASES}")

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_VAL,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False,
    )
    class_names = list(val_ds.class_names)
    print(f"-> Orden de carpetas detectado: {class_names}")
    if class_names != CLASES:
        print("ADVERTENCIA: el orden de carpetas no coincide con clases.json")

    print(f"-> Cargando modelo: {MODEL_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH))

    loss, accuracy = model.evaluate(val_ds, verbose=0)
    print(f"\n>> Loss: {loss:.4f}")
    print(f">> Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    y_true, y_pred, y_prob = [], [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_prob.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(np.argmax(preds, axis=1).tolist())

    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=CLASES,
        zero_division=0,
        output_dict=True,
    )
    report_txt = classification_report(
        y_true, y_pred, target_names=CLASES, zero_division=0
    )
    print("\n" + report_txt)

    per_class = {
        clase: {
            "precision": round(report_dict[clase]["precision"], 4),
            "recall": round(report_dict[clase]["recall"], 4),
            "f1_score": round(report_dict[clase]["f1-score"], 4),
            "support": int(report_dict[clase]["support"]),
        }
        for clase in CLASES
    }

    resultados = {
        "modelo": MODEL_PATH.name,
        "epocas_entrenamiento": 35,
        "learning_rate": 0.001,
        "umbral_requerido": UMBRAL,
        "loss_validacion": round(float(loss), 4),
        "accuracy_validacion": round(float(accuracy), 4),
        "accuracy_porcentaje": round(float(accuracy) * 100, 2),
        "cumple_umbral": bool(accuracy >= UMBRAL),
        "total_validacion": int(len(y_true)),
        "matriz_confusion": cm.tolist(),
        "clases": CLASES,
        "metricas_por_clase": per_class,
        "macro_avg": {
            "precision": round(report_dict["macro avg"]["precision"], 4),
            "recall": round(report_dict["macro avg"]["recall"], 4),
            "f1_score": round(report_dict["macro avg"]["f1-score"], 4),
        },
        "weighted_avg": {
            "precision": round(report_dict["weighted avg"]["precision"], 4),
            "recall": round(report_dict["weighted avg"]["recall"], 4),
            "f1_score": round(report_dict["weighted avg"]["f1-score"], 4),
        },
    }

    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)
    print(f"-> Resultados guardados en: {RESULTS_JSON_PATH}")

    # Grafica comparativa por clase (tercera evidencia visual)
    x = np.arange(len(CLASES))
    width = 0.26
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, [per_class[c]["precision"] for c in CLASES], width, label="Precision")
    plt.bar(x, [per_class[c]["recall"] for c in CLASES], width, label="Recall")
    plt.bar(x + width, [per_class[c]["f1_score"] for c in CLASES], width, label="F1-score")
    plt.xticks(x, CLASES, rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Puntaje")
    plt.title("Metricas por Genero Musical - Conjunto de Validacion")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PER_CLASS_PLOT_PATH, dpi=200)
    plt.close()
    print(f"-> Grafica por clase guardada en: {PER_CLASS_PLOT_PATH}")

    if not resultados["cumple_umbral"]:
        print(
            f"\n[!] Accuracy por debajo del {UMBRAL * 100:.0f}%. "
            "Ajustar LEARNING_RATE a 0.001 y reentrenar."
        )
    else:
        print(f"\n[OK] Accuracy supera el umbral del {UMBRAL * 100:.0f}%.")


if __name__ == "__main__":
    main()
