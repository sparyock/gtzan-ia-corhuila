import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os

BATCH_SIZE = 32
IMG_SIZE = (128, 128)
EPOCHS = 20
CLASES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

def cargar_datos():
    print("Cargando imágenes procesadas...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset/procesado/train',
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=True
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        'dataset/procesado/val',
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    )
    return train_ds, val_ds

def construir_modelo():
    model = models.Sequential([
        # Capa explícita de entrada
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        
        # Normalización
        layers.Rescaling(1./255),
        
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Capas Densas
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generar_graficas(history, model, val_ds):
    os.makedirs('web_app/static', exist_ok=True)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_range, acc, label='Train Accuracy')
    ax1.plot(epochs_range, val_acc, label='Val Accuracy')
    ax1.set_title('Precisión del Modelo')
    ax1.legend(loc='lower right')

    ax2.plot(epochs_range, loss, label='Train Loss')
    ax2.plot(epochs_range, val_loss, label='Val Loss')
    ax2.set_title('Pérdida del Modelo')
    ax2.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('web_app/static/training_plot.png')
    
    y_true = []
    y_pred = []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASES, yticklabels=CLASES)
    plt.title('Matriz de Confusión')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig('web_app/static/confusion_matrix.png')

if __name__ == "__main__":
    train_ds, val_ds = cargar_datos()
    modelo = construir_modelo()
    modelo.summary()
    
    print("\nIniciando entrenamiento reestructurado...")
    historial = modelo.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    os.makedirs('web_app/model', exist_ok=True)
    modelo.save('web_app/model/modelo_final.keras')
    print("\nModelo guardado en web_app/model/modelo_final.keras")
    
    print("\nGenerando gráficas de evaluación...")
    generar_graficas(historial, modelo, val_ds)
    print("¡Entrenamiento finalizado con éxito!")