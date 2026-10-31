import os
import shutil
from sklearn.model_selection import train_test_split

ORIGEN = 'dataset/images_original'
DESTINO = 'dataset/procesado'
generos = ['blues', 'classical', 'country', 'disco', 'hiphop', 
            'metal', 'pop', 'reggae', 'rock']

# Crear carpetas de salida
for particion in ['train', 'val']:
    for g in generos:
        os.makedirs(os.path.join(DESTINO, particion, g), exist_ok=True)

print("Organizando imágenes en Train (80%) y Val (20%)...")
for g in generos:
    ruta_genero = os.path.join(ORIGEN, g)
    if not os.path.exists(ruta_genero):
        print(f"Advertencia: No se encontró la carpeta {ruta_genero}")
        continue
        
    imagenes = [f for f in os.listdir(ruta_genero) if f.endswith('.png')]
    train_imgs, val_imgs = train_test_split(imagenes, test_size=0.20, random_state=42)
    
    for img in train_imgs:
        shutil.copy(os.path.join(ruta_genero, img), os.path.join(DESTINO, 'train', g, img))
    for img in val_imgs:
        shutil.copy(os.path.join(ruta_genero, img), os.path.join(DESTINO, 'val', g, img))

print("¡Listo! Las imágenes se organizaron en dataset/procesado/")