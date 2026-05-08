import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.data.loader import get_data_generators
from src.utils.config import MODEL_PATH, CLASSES

def evaluate():
    _, val_gen = get_data_generators()

    print(f"🔄 Cargando modelo desde {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("📊 Evaluando sobre conjunto de validación...\n")
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    print("=== Reporte de Clasificación ===")
    print(classification_report(y_true, y_pred, target_names=CLASSES))

    print("=== Matriz de Confusión ===")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

if __name__ == '__main__':
    evaluate()