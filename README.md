# 🛣️ Road Damage Classifier
### Sistema de Clasificación de Daños en Carreteras mediante Visión Artificial

Proyecto desarrollado para la asignatura **Diseño de Sistemas de Visión por Computador (DVC101)**  
Universidad Don Bosco — Facultad de Ingeniería

---

## 1. Descripción del Proyecto

Sistema de clasificación de imágenes basado en **Transfer Learning (MobileNetV2)** capaz de identificar automáticamente el tipo de daño presente en una imagen de carretera.

El sistema clasifica imágenes en 4 categorías:

| Clase | Descripción |
|---|---|
| `pothole` | Bache |
| `longitudinal_crack` | Grieta longitudinal |
| `transverse_crack` | Grieta transversal |
| `alligator_crack` | Grieta en cocodrilo |

**Pipeline del sistema:**
```
Imagen de entrada
      ↓
Preprocesamiento (resize 224x224, normalización)
      ↓
MobileNetV2 (Transfer Learning + Fine-tuning)
      ↓
Clasificador (Dense 256 → Dense 128 → Softmax 4 clases)
      ↓
Clase predicha + Nivel de confianza
```

**Dataset:** Road Damage Dataset (Kaggle) — 3,321 imágenes reales, split 80/20 train/val  
**Framework:** TensorFlow 2.15 / Keras  
**Lenguaje:** Python 3.10

---

## 2. Dataset

Debido al tamaño del dataset, las imágenes originales no se incluyen en este repositorio.

El proyecto utiliza el dataset público:

**Road Damage Dataset (YOLO format)**  
Autor: `alvarobasily` en Kaggle

### Descarga del dataset

Descargar desde:

https://www.kaggle.com/datasets/alvarobasily/road-damage

### Estructura esperada

Después de descargar el dataset, colocar los archivos en:

```bash
data/raw/
```

La carpeta debe contener imágenes y archivos `.txt` en formato YOLO:

```text
data/raw/
├── image_001.jpg
├── image_001.txt
├── image_002.jpg
├── image_002.txt
└── ...
```

Cada archivo `.txt` contiene anotaciones en formato:

```text
class_id x_center y_center width height
```

### Organización automática del dataset

El proyecto incluye un script para organizar automáticamente las imágenes por clase y generar el split train/validation (80/20):

```bash
python organize_dataset.py
```

Esto generará automáticamente:

```text
data/processed/
├── train/
│   ├── pothole/
│   ├── longitudinal_crack/
│   ├── transverse_crack/
│   └── alligator_crack/
└── val/
    ├── pothole/
    ├── longitudinal_crack/
    ├── transverse_crack/
    └── alligator_crack/
```

### Notas importantes

- El dataset original contiene anotaciones en formato YOLO.
- El script `organize_dataset.py` selecciona la clase dominante de cada imagen.
- El split se realiza de forma reproducible usando `random seed = 42`.
- El modelo entrenado `.keras` sí está incluido en este repositorio y puede utilizarse directamente para inferencia sin necesidad de reentrenamiento.

---

## 3. Cómo Ejecutar

## Requisitos previos
- Python 3.10
- Anaconda (recomendado)

## Instalación del entorno

```bash
# 1. Crear entorno conda
conda create -n road_damage python=3.10 -y
conda activate road_damage

# 2. Instalar dependencias
pip install -r requirements.txt
```

## Inferencia (uso principal)

```bash
# Activar entorno
conda activate road_damage

# Ejecutar inferencia sobre una imagen
python scripts/run_inference.py --input ruta/a/imagen.jpg
```

> ⚠️ **El modelo ya está entrenado y guardado en `models/`.**  
> No se requiere reentrenamiento para evaluar el sistema.

## 2. Cómo Ejecutar

### Requisitos previos
- Python 3.10
- Anaconda (recomendado)

### Instalación del entorno

```bash
# 1. Crear entorno conda
conda create -n road_damage python=3.10 -y
conda activate road_damage

# 2. Instalar dependencias
pip install -r requirements.txt
```

### Inferencia (uso principal)

```bash
# Activar entorno
conda activate road_damage

# Ejecutar inferencia sobre una imagen
python scripts/run_inference.py --input ruta/a/imagen.jpg
```

> ⚠️ **El modelo ya está entrenado y guardado en `models/`.**  
> No se requiere reentrenamiento para evaluar el sistema.

### Argumentos disponibles

```
--input     Ruta a la imagen de entrada (.jpg / .jpeg / .png)  [requerido]
```

---

## 4. Estructura del Proyecto

```
road_damage_classifier/
│
├── data/
│   ├── raw/                        ← Imágenes originales del dataset
│   └── processed/
│       ├── train/                  ← 2,655 imágenes para entrenamiento (80%)
│       │   ├── alligator_crack/
│       │   ├── longitudinal_crack/
│       │   ├── pothole/
│       │   └── transverse_crack/
│       └── val/                    ← 666 imágenes para validación (20%)
│           ├── alligator_crack/
│           ├── longitudinal_crack/
│           ├── pothole/
│           └── transverse_crack/
│
├── models/
│   └── road_damage_model.keras     ← Modelo entrenado (no requiere reentrenamiento)
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── loader.py               ← Generadores de datos con data augmentation
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── model_builder.py        ← Arquitectura MobileNetV2 + capas personalizadas
│   │   ├── train.py                ← Entrenamiento + fine-tuning
│   │   └── evaluate.py             ← Métricas: accuracy, precision, recall, F1
│   ├── inference/
│   │   └── predict.py              ← Lógica de inferencia (CRÍTICO)
│   └── utils/
│       ├── config.py               ← Parámetros centrales del sistema
│       └── logger.py
│
├── scripts/
│   └── run_inference.py            ← EJECUTABLE PRINCIPAL ← ejecutar aquí
│
├── organize_dataset.py             ← Script de organización del dataset
├── requirements.txt                ← Dependencias del proyecto
└── README.md
```

---

## 5. Ejemplo de Uso

### Caso 1 — Imagen con bache
```bash
python scripts/run_inference.py --input data/raw/A_2890.jpeg
```
```
🔄 Cargando modelo...
📸 Procesando: data/raw/A_2890.jpeg

========== RESULTADO ==========
Input:       data/raw/A_2890.jpeg
Prediction:  longitudinal_crack
Confidence:  38.36%
================================
```

### Caso 2 — Imagen propia
```bash
python scripts/run_inference.py --input C:/Users/TuUsuario/fotos/carretera.jpg
```
```
🔄 Cargando modelo...
📸 Procesando: C:/Users/TuUsuario/fotos/carretera.jpg

========== RESULTADO ==========
Input:       C:/Users/TuUsuario/fotos/carretera.jpg
Prediction:  pothole
Confidence:  74.20%
================================
```

### Evaluación completa con métricas
```bash
python -m src.models.evaluate
```

---

## 6. Resultados Esperados

### Métricas del modelo final

El sistema fue entrenado en 2 fases:

| Fase | Descripción | Val Accuracy |
|---|---|---|
| Fase 1 | Transfer Learning — capas base congeladas | 59.2% |
| Fase 2 | Fine-tuning — últimas 30 capas descongeladas | **63.4%** |

### Reporte de clasificación (conjunto de validación — 666 imágenes)

| Clase | Precision | Recall | F1-score | Soporte |
|---|---|---|---|---|
| alligator_crack | 0.59 | 0.52 | 0.56 | 164 |
| longitudinal_crack | 0.52 | 0.47 | 0.49 | 123 |
| pothole | **0.75** | **0.73** | **0.74** | 251 |
| transverse_crack | 0.57 | 0.73 | 0.64 | 128 |
| **Macro avg** | **0.61** | **0.62** | **0.61** | 666 |
| **Accuracy global** | | | **0.63** | 666 |

### Análisis de resultados

- **Pothole** es la clase con mejor desempeño (F1: 0.74) porque los baches tienen características visuales claramente distintas al resto.
- **Longitudinal crack** es la clase más difícil (F1: 0.49) debido a su similitud visual con grietas transversales y en cocodrilo.
- El modelo fue entrenado **exclusivamente en CPU**, lo cual limita el número de épocas exploradas. Con GPU se esperarían mejoras de 10–15% en accuracy.

### Limitaciones identificadas

- Dataset sin clase "sin daño" — el modelo no puede clasificar carreteras en buen estado
- Sensibilidad a condiciones de iluminación
- Confusión entre tipos de grietas visualmente similares

---

## Tecnologías Utilizadas

| Componente | Tecnología |
|---|---|
| Lenguaje | Python 3.10 |
| Framework ML | TensorFlow 2.15 / Keras |
| Modelo base | MobileNetV2 (ImageNet) |
| Procesamiento de imágenes | OpenCV, Pillow |
| Métricas | scikit-learn |
| Dataset | Road Damage Dataset (Kaggle — alvarobasily) |
