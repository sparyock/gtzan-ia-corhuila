import os
import json
import uuid
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request
import tensorflow as tf
import librosa
import librosa.display
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / 'uploads'
STATIC_SPECS = BASE_DIR / 'static' / 'specs'
MODEL_PATH = BASE_DIR / 'model' / 'modelo_final.keras'
CLASSES_PATH = BASE_DIR / 'model' / 'clases.json'

UPLOAD_FOLDER.mkdir(exist_ok=True)
STATIC_SPECS.mkdir(parents=True, exist_ok=True)

# Cargar modelo y clases
print("Cargando modelo...")
model = tf.keras.models.load_model(str(MODEL_PATH))
print("Modelo cargado exitosamente")

with open(CLASSES_PATH, 'r', encoding='utf-8') as f:
    clases_dict = json.load(f)

CLASES = [clases_dict[str(i)] for i in range(len(clases_dict))]
print(f"Clases cargadas: {CLASES}")

IMG_SIZE = (128, 128)

def audio_to_spectrogram(audio_path):
    """Convierte archivo de audio WAV a espectrograma Mel 128x128 en escala de grises"""
    y, sr = librosa.load(audio_path, duration=30)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis('off')
    librosa.display.specshow(S_dB, sr=sr, cmap='magma')
    plt.tight_layout(pad=0)

    spec_filename = f'spectrogram_{uuid.uuid4().hex}.png'
    spec_path = STATIC_SPECS / spec_filename
    plt.savefig(spec_path, bbox_inches='tight', pad_inches=0, transparent=False)
    plt.close(fig)

    img = Image.open(spec_path).convert('L')
    img_resized = img.resize(IMG_SIZE)

    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, f'specs/{spec_filename}'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'audio' not in request.files:
            return render_template('index.html', error='No se seleccionó ningún archivo')

        file = request.files['audio']
        if file.filename == '':
            return render_template('index.html', error='No se seleccionó ningún archivo')

        if not file.filename.lower().endswith('.wav'):
            return render_template('index.html', error='Solo se permiten archivos .WAV')

        filename = file.filename
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))

        img_array, spec_url = audio_to_spectrogram(str(filepath))

        predictions = model.predict(img_array, verbose=0)[0]
        top_indices = predictions.argsort()[-3:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                'genre': CLASES[idx],
                'confidence': round(float(predictions[idx]) * 100, 2)
            })

        try:
            os.remove(filepath)
        except:
            pass

        return render_template('index.html', results=results, spec_image=spec_url)

    except Exception as e:
        return render_template('index.html', error=f'Error al procesar el audio: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
