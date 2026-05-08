"""
Script para organizar el dataset Road Damage en carpetas por clase.
Lee archivos de anotación en formato YOLO (.txt) donde cada línea es:
   class_id x_center y_center width height
y copia cada imagen a la carpeta de su clase dominante (train/val 80/20).
"""

import os
import shutil
import random
from collections import defaultdict

# ── Configuración ──────────────────────────────────────────────
RAW_DIR       = 'data/raw'
PROCESSED_DIR = 'data/processed'
TRAIN_RATIO   = 0.80
RANDOM_SEED   = 42

# Mapeo: class_id del txt → nombre de carpeta
# Orden según el dataset alvarobasily/road-damage (YOLO format)
CLASS_MAP = {
    0: 'pothole',
    1: 'longitudinal_crack',
    2: 'transverse_crack',
    3: 'alligator_crack',
}
# ───────────────────────────────────────────────────────────────

def get_class_from_txt(txt_path):
    """
    Lee el archivo YOLO .txt y retorna la clase más frecuente.
    Cada línea tiene formato: class_id x y w h
    """
    try:
        with open(txt_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if not lines:
            return None
        ids = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 1:
                try:
                    class_id = int(parts[0])
                    if class_id in CLASS_MAP:
                        ids.append(class_id)
                except ValueError:
                    continue
        if ids:
            return max(set(ids), key=ids.count)
    except Exception as e:
        print(f"  ⚠️  Error leyendo {txt_path}: {e}")
    return None

def create_folder_structure():
    for split in ['train', 'val']:
        for cls in CLASS_MAP.values():
            os.makedirs(os.path.join(PROCESSED_DIR, split, cls), exist_ok=True)
    print("✅ Estructura de carpetas creada.\n")

def organize():
    create_folder_structure()

    txt_files = [f for f in os.listdir(RAW_DIR) if f.endswith('.txt')]
    print(f"📂 Leyendo {len(txt_files)} archivos .txt en data/raw/ ...\n")

    class_images = defaultdict(list)

    for txt_file in txt_files:
        txt_path = os.path.join(RAW_DIR, txt_file)
        class_id = get_class_from_txt(txt_path)
        if class_id is None:
            continue

        base_name = os.path.splitext(txt_file)[0]
        for ext in ['.jpeg', '.jpg', '.png']:
            img_path = os.path.join(RAW_DIR, base_name + ext)
            if os.path.exists(img_path):
                class_images[class_id].append(img_path)
                break

    # Muestra conteo por clase
    print("📊 Imágenes encontradas por clase:")
    for cid, imgs in sorted(class_images.items()):
        print(f"   [{cid}] {CLASS_MAP[cid]:<25} → {len(imgs)} imágenes")

    if not class_images:
        print("\n❌ No se encontraron imágenes. Revisa que data/raw/ tenga los archivos.")
        return

    # Split y copia
    random.seed(RANDOM_SEED)
    total_train, total_val = 0, 0

    for class_id, images in class_images.items():
        folder = CLASS_MAP[class_id]
        random.shuffle(images)
        split_idx  = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        for img in train_imgs:
            dst = os.path.join(PROCESSED_DIR, 'train', folder, os.path.basename(img))
            shutil.copy2(img, dst)

        for img in val_imgs:
            dst = os.path.join(PROCESSED_DIR, 'val', folder, os.path.basename(img))
            shutil.copy2(img, dst)

        total_train += len(train_imgs)
        total_val   += len(val_imgs)

    print(f"\n✅ Dataset organizado exitosamente:")
    print(f"   Train : {total_train} imágenes")
    print(f"   Val   : {total_val}   imágenes")
    print(f"   Total : {total_train + total_val} imágenes")
    print(f"\n📁 Resultado en: {PROCESSED_DIR}/")

if __name__ == '__main__':
    organize()