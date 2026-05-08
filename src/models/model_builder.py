import tensorflow as tf
from src.utils.config import IMG_SIZE, NUM_CLASSES, LEARNING_RATE

def build_model():
    """
    Transfer Learning con MobileNetV2 preentrenado en ImageNet.

    Estrategia:
    - Capas base CONGELADAS: no se reentrenan, solo extraen features.
    - Se agrega cabeza de clasificación personalizada para 4 clases.
    - Dropout 0.4 para reducir overfitting dado el tamaño del dataset.
    """
    # 1. Modelo base preentrenado (sin la cabeza de clasificación de ImageNet)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # 2. Congelar todas las capas base
    base_model.trainable = False

    # 3. Construir modelo completo
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = base_model(inputs, training=False)          # training=False = inferencia en BatchNorm
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    # 4. Compilar
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model