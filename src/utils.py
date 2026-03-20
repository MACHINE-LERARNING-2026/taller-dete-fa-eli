"""
UTILIDADES – Taller YOLO Detección de Casas
============================================
Funciones auxiliares para:
  - Dibujar bounding boxes sobre imágenes
  - Convertir formatos de anotación
  - Visualizar resultados
  - Parsear resultados del modelo YOLO
"""

import cv2
import numpy as np
from PIL import Image


# =====================================================
# CONSTANTES DE VISUALIZACIÓN
# =====================================================

# Color principal para las cajas (verde lima, visible sobre mayoría de fondos)
COLOR_CAJA          = (0, 200, 80)     # BGR para OpenCV
COLOR_TEXTO         = (255, 255, 255)  # Blanco
COLOR_FONDO_TEXTO   = (0, 200, 80)     # Mismo verde para el fondo de la etiqueta
GROSOR_CAJA         = 2                # Píxeles de grosor de la caja
FUENTE_CV2          = cv2.FONT_HERSHEY_SIMPLEX
ESCALA_FUENTE       = 0.6
GROSOR_FUENTE       = 2


# =====================================================
# SECCIÓN 1: DIBUJO Y VISUALIZACIÓN
# =====================================================

def dibujar_detecciones(
    imagen: np.ndarray,
    cajas: list[list[float]],
    scores: list[float],
    clases: list[str],
    umbral_confianza: float = 0.25,
) -> np.ndarray:
    """
    Dibuja bounding boxes y etiquetas sobre una imagen NumPy (BGR).

    Parámetros
    ----------
    imagen             : Imagen en formato NumPy BGR (como la entrega OpenCV).
    cajas              : Lista de cajas en formato [x1, y1, x2, y2] (píxeles absolutos).
    scores             : Lista de confianzas (0.0 – 1.0) para cada caja.
    clases             : Lista de nombres de clase para cada caja.
    umbral_confianza   : Mínima confianza para mostrar una detección.

    Retorna
    -------
    Imagen anotada como NumPy BGR (copia, no modifica la original).
    """
    imagen_anotada = imagen.copy()

    for caja, score, clase in zip(cajas, scores, clases):

        # Saltar detecciones por debajo del umbral de confianza
        if score < umbral_confianza:
            continue

        x1, y1, x2, y2 = map(int, caja)

        # ── Dibujar rectángulo de la caja ──────────────────────────
        cv2.rectangle(
            imagen_anotada,
            (x1, y1),
            (x2, y2),
            COLOR_CAJA,
            GROSOR_CAJA,
        )

        # ── Preparar texto de etiqueta ─────────────────────────────
        etiqueta = f"{clase}: {score:.2f}"
        (ancho_txt, alto_txt), baseline = cv2.getTextSize(
            etiqueta, FUENTE_CV2, ESCALA_FUENTE, GROSOR_FUENTE
        )

        # Fondo sólido para la etiqueta (mejora la legibilidad)
        y_fondo = max(y1 - alto_txt - baseline - 4, 0)
        cv2.rectangle(
            imagen_anotada,
            (x1, y_fondo),
            (x1 + ancho_txt + 4, y1),
            COLOR_FONDO_TEXTO,
            thickness=-1,  # Relleno sólido
        )

        # Texto de la etiqueta
        cv2.putText(
            imagen_anotada,
            etiqueta,
            (x1 + 2, y1 - baseline - 2),
            FUENTE_CV2,
            ESCALA_FUENTE,
            COLOR_TEXTO,
            GROSOR_FUENTE,
            lineType=cv2.LINE_AA,
        )

    return imagen_anotada


def dibujar_conteo_umbral(imagen: np.ndarray,cantidad: int,umbral: float) -> np.ndarray:
    """
    Añade un contador de casas detectadas y el umbral de confianza
    en la esquina inferior izquierda.

    Parámetros
    ----------
    imagen   : Imagen NumPy BGR.
    cantidad : Número de casas detectadas.
    umbral   : Umbral de confianza utilizado en la inferencia.

    Retorna
    -------
    Imagen con el contador y umbral superpuestos.
    """
    imagen_anotada = imagen.copy()

    alto_img, ancho_img = imagen_anotada.shape[:2]

    texto = f"Casas: {cantidad} | Umbral: {umbral:.2f}"

    # Obtener tamaño del texto dinámicamente
    (ancho_txt, alto_txt), baseline = cv2.getTextSize(
        texto,
        FUENTE_CV2,
        0.7,
        2
    )

    # Coordenadas esquina inferior izquierda
    x_inicio = 10
    y_inicio = alto_img - 10

    # Fondo semitransparente
    overlay = imagen_anotada.copy()
    cv2.rectangle(
        overlay,
        (x_inicio - 5, y_inicio - alto_txt - baseline - 10),
        (x_inicio + ancho_txt + 5, y_inicio + 5),
        (0, 0, 0),
        -1
    )

    cv2.addWeighted(overlay, 0.5, imagen_anotada, 0.5, 0, imagen_anotada)

    # Dibujar texto
    cv2.putText(
        imagen_anotada,
        texto,
        (x_inicio, y_inicio),
        FUENTE_CV2,
        0.7,
        COLOR_TEXTO,
        2,
        cv2.LINE_AA,
    )

    return imagen_anotada


# =====================================================
# SECCIÓN 2: CONVERSIÓN DE FORMATOS
# =====================================================

def numpy_a_bytes(imagen_bgr: np.ndarray, extension: str = ".jpg") -> bytes:
    """
    Convierte una imagen NumPy BGR a bytes para respuestas HTTP.

    Parámetros
    ----------
    imagen_bgr : Imagen en formato NumPy BGR.
    extension  : Formato de salida ('.jpg' o '.png').

    Retorna
    -------
    Bytes de la imagen codificada.
    """
    exito, buffer = cv2.imencode(extension, imagen_bgr)
    if not exito:
        raise ValueError(f"No se pudo codificar la imagen en formato {extension}")
    return buffer.tobytes()


def bytes_a_numpy(datos: bytes) -> np.ndarray:
    """
    Convierte bytes de imagen a NumPy BGR (formato OpenCV).

    Parámetros
    ----------
    datos : Bytes crudos de la imagen.

    Retorna
    -------
    Imagen en NumPy BGR.
    """
    arreglo = np.frombuffer(datos, dtype=np.uint8)
    imagen = cv2.imdecode(arreglo, cv2.IMREAD_COLOR)

    if imagen is None:
        raise ValueError(
            "No se pudo decodificar la imagen desde los bytes proporcionados"
        )

    return imagen


# =====================================================
# SECCIÓN 3: PARSEO DE RESULTADOS ULTRALYTICS
# =====================================================

def parsear_resultados_yolo(resultado) -> dict:
    """
    Extrae cajas, scores y clases del objeto Results de Ultralytics.

    Parámetros
    ----------
    resultado : Objeto Results de Ultralytics (un solo frame).

    Retorna
    -------
    Diccionario con listas de cajas, scores, clases e índices de clase.
    """
    cajas_xyxy  = []
    scores      = []
    clases      = []
    ids_clase   = []

    # Iterar sobre cada detección encontrada
    for caja in resultado.boxes:

        # Coordenadas absolutas [x1, y1, x2, y2]
        coords = caja.xyxy[0].tolist()
        cajas_xyxy.append(coords)

        # Confianza de la detección
        scores.append(float(caja.conf[0]))

        # Nombre e índice de la clase detectada
        idx_clase = int(caja.cls[0])
        ids_clase.append(idx_clase)

        nombre_clase = resultado.names.get(idx_clase, f"clase_{idx_clase}")
        clases.append(nombre_clase)

    return {
        "cajas_xyxy": cajas_xyxy,
        "scores": scores,
        "clases": clases,
        "ids_clase": ids_clase,
        "total": len(cajas_xyxy),
    }