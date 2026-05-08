import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import tensorflow as tf
from PIL import Image

from src.utils.config import MODEL_PATH, IMG_SIZE, CLASSES

def load_model():
    """Carga el modelo entrenado desde disco. No requiere reentrenamiento."""
    return tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_path):
    """Preprocesa una imagen para el modelo: resize + normalización."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)   # (1, 224, 224, 3)

def predict(image_path, model):
    """Retorna la clase predicha y su nivel de confianza."""
    img    = preprocess_image(image_path)
    probs  = model.predict(img, verbose=0)
    idx    = np.argmax(probs)
    label  = CLASSES[idx]
    confidence = float(probs[0][idx])
    return label, confidence