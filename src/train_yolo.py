"""
train_yolo.py
-------------------------------------------------------
Script de entrenamiento para el modelo YOLO – Detección de Casas

Estructura esperada del proyecto:

taller-yolo-casas/
│
├── src/
├── models/
├── data.yaml
└── ...

Este script:
1. Entrena el modelo YOLO.
2. Guarda resultados en models/detect/.
3. Copia el best.pt final como models/house-yolo.pt.
"""

import os
import shutil
from ultralytics import YOLO


# =====================================================
# CONFIGURACIÓN
# =====================================================

# Rutas base del proyecto
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML_PATH = os.path.join(ROOT_DIR, "data.yaml")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DETECT_DIR = os.path.join(MODELS_DIR, "detect")

# Parámetros de entrenamiento
WEIGHTS_DIR = os.path.join(MODELS_DIR, "weights")
MODEL_BASE = os.path.join(WEIGHTS_DIR, "yolov8m-seg.pt")
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16


# =====================================================
# FUNCIONES
# =====================================================

def train() -> None:
    """
    Ejecuta el entrenamiento del modelo YOLO.
    """

    print("Iniciando entrenamiento YOLO...")

    model = YOLO(MODEL_BASE)

    results = model.train(
        data=DATA_YAML_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        project=MODELS_DIR,
        name="detect",
        exist_ok=True
    )

    print("✅ Entrenamiento finalizado.")
    save_best_model(results)


def save_best_model(results) -> None:
    """
    Copia el best.pt generado por YOLO a:
    models/house-yolo.pt
    """

    best_model_path = os.path.join(
        results.save_dir,
        "weights",
        "best.pt"
    )

    final_model_path = os.path.join(MODELS_DIR, "house-yolo.pt")

    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, final_model_path)
        print(f"🏆 Modelo final guardado en: {final_model_path}")
    else:
        print("⚠️ No se encontró el archivo best.pt")


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    # Crear carpeta models si no existe
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    train()

    print("Proceso completado correctamente.")