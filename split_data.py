import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
ORIGEN = BASE_DIR / 'dataset' / 'images_original'
DESTINO = BASE_DIR / 'dataset' / 'procesado'
EXTENSIONES = ('.png', '.jpg', '.jpeg')
generos = ['blues', 'classical', 'country', 'disco', 'hiphop',
           'metal', 'pop', 'reggae', 'rock']

# Crear carpetas de salida
for particion in ['train', 'val']:
    for g in generos:
        (DESTINO / particion / g).mkdir(parents=True, exist_ok=True)

print("Organizando imágenes en Train (80%) y Val (20%)...")
for g in generos:
    ruta_genero = ORIGEN / g
    if not ruta_genero.exists():
        print(f"Advertencia: No se encontró la carpeta {ruta_genero}")
        continue

    imagenes = [f.name for f in ruta_genero.iterdir()
                if f.is_file() and f.suffix.lower() in EXTENSIONES]
    if not imagenes:
        print(f"Advertencia: No hay imágenes en {ruta_genero}")
        continue

    train_imgs, val_imgs = train_test_split(imagenes, test_size=0.20, random_state=42)

    for img in train_imgs:
        shutil.copy(ruta_genero / img, DESTINO / 'train' / g / img)
    for img in val_imgs:
        shutil.copy(ruta_genero / img, DESTINO / 'val' / g / img)
    print(f"  {g}: {len(train_imgs)} train / {len(val_imgs)} val")

print("¡Listo! Las imágenes se organizaron en dataset/procesado/")