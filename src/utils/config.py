# Configuración central del proyecto
IMG_SIZE = (224, 224)        # Tamaño requerido por MobileNetV2
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
NUM_CLASSES = 4              

CLASSES = ['alligator_crack', 'longitudinal_crack', 'pothole', 'transverse_crack']

TRAIN_DIR  = 'data/processed/train'
VAL_DIR    = 'data/processed/val'
MODEL_PATH = 'models/road_damage_model.keras'