import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import os
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from src.data.loader import get_data_generators
from src.models.model_builder import build_model
from src.utils.config import EPOCHS, MODEL_PATH

def compute_weights(train_gen):
    """
    Calcula pesos por clase corregidos para que las claves
    vayan exactamente de 0 a NUM_CLASSES-1.
    """
    labels  = train_gen.classes
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    # Remapea índices a 0..N-1 sin importar lo que devuelva sklearn
    weight_dict = {}
    for i, w in enumerate(weights):
        weight_dict[i] = w
    print(f"   Clases encontradas : {list(classes)}")
    print(f"   Pesos calculados   : {weight_dict}")
    return weight_dict

def train():
    os.makedirs('models', exist_ok=True)

    train_gen, val_gen = get_data_generators()
    model = build_model()

    print("\n=== Arquitectura del modelo ===")
    model.summary()

    # Pesos para manejar desbalance de clases
    class_weights = compute_weights(train_gen)
    print(f"\n⚖️  Pesos por clase: {class_weights}")

    # Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    print("\n=== Iniciando entrenamiento ===\n")

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    print(f"\n✅ Modelo guardado en: {MODEL_PATH}")
    return history
def fine_tune():
    """
    Fase 2: descongelar las últimas 30 capas de MobileNetV2 y
    reentrenar con learning rate muy pequeño para refinar features.
    """
    os.makedirs('models', exist_ok=True)

    train_gen, val_gen = get_data_generators()
    class_weights = compute_weights(train_gen)

    # Cargar el mejor modelo guardado en fase 1
    print(f"🔄 Cargando modelo desde {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # Descongelar las últimas 30 capas del base model
    base_model = model.layers[1]
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"   Capas entrenables ahora: {trainable}")

    # Recompilar con learning rate mucho más pequeño
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print("\n=== Iniciando fine-tuning ===\n")
    model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stop]
    )

    print(f"\n✅ Modelo refinado guardado en: {MODEL_PATH}")
if __name__ == '__main__':
    fine_tune()