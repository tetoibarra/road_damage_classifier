import tensorflow as tf
from src.utils.config import IMG_SIZE, BATCH_SIZE, TRAIN_DIR, VAL_DIR


def get_data_generators():
    """
    Crea generadores de imágenes.
    Train: con data augmentation para mejorar generalización.
    Val:   sin augmentation para evaluación limpia.
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2]
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen