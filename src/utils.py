"""
UTILIDADES – Taller YOLO Detección de Casas
============================================
Funciones auxiliares para:
  - Dibujar bounding boxes sobre imágenes
  - Dibujar máscaras de segmentación
  - Convertir formatos de anotación
  - Visualizar resultados
  - Parsear resultados del modelo YOLO (detección y segmentación)
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

# Colores de máscara por índice de clase (BGR)
# Índice 0 = Fachada → azul semitransparente
# Índice 1 = Poste   → rojo semitransparente
COLOR_MASCARA: dict[int, tuple[int, int, int]] = {
    0: (200, 80,  0),   # Fachada – azul
    1: (0,   80, 200),  # Poste   – rojo
}
ALPHA_MASCARA = 0.45  # Opacidad del overlay de máscara


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


def generar_mascara_postes(
    mascaras: list[np.ndarray],
    ids_clase: list[int],
    shape: tuple[int, int],
    dilation_px: int = 20,
) -> np.ndarray:
    """
    Combina todas las máscaras de postes en una única imagen binaria
    y aplica dilatación para mejorar el resultado del inpainting.

    LaMa reconstruye mejor cuando la máscara cubre ligeramente más
    allá del borde del objeto — la dilatación logra ese efecto.

    El resultado es una imagen en escala de grises donde:
      - 255 (blanco) = región a eliminar (poste + margen dilatado)
      - 0   (negro)  = fondo a conservar

    Parámetros
    ----------
    mascaras    : Lista de arrays binarios (H, W) uint8 — uno por detección.
    ids_clase   : Lista de índices de clase para cada máscara.
    shape       : Tupla (alto, ancho) de la imagen original.
    dilation_px : Radio de dilatación en píxeles (por defecto 20).

    Retorna
    -------
    Array (H, W) uint8 con 255 en la región de postes dilatada y 0 en el resto.
    """
    alto, ancho = shape
    mascara_combinada = np.zeros((alto, ancho), dtype=np.uint8)

    for mask, id_clase in zip(mascaras, ids_clase):
        if id_clase == 1:  # Poste
            mascara_combinada = np.maximum(mascara_combinada, mask * 255)

    if mascara_combinada.max() > 0 and dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1, dilation_px * 2 + 1)
        )
        mascara_combinada = cv2.dilate(mascara_combinada, kernel)

    return mascara_combinada


def armar_collage(
    imagen_deteccion: np.ndarray,
    mascara: np.ndarray,
    imagen_resultado: np.ndarray,
    alto_etiqueta: int = 30,
) -> np.ndarray:
    """
    Arma un collage horizontal con las tres imágenes del pipeline:
    detección | máscara | resultado sin postes.

    Parámetros
    ----------
    imagen_deteccion : Imagen BGR con máscaras y bounding boxes dibujados.
    mascara          : Máscara binaria (H, W) uint8 de los postes.
    imagen_resultado : Imagen BGR con los postes eliminados por inpainting.
    alto_etiqueta    : Altura en píxeles de la barra de título de cada panel.

    Retorna
    -------
    Imagen BGR con los tres paneles lado a lado y etiquetas en la parte superior.
    """
    etiquetas = ["Detección", "Máscara postes", "Sin postes (LaMa)"]

    # Convertir máscara gris a BGR para poder concatenar con las otras
    mascara_bgr = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)

    paneles = [imagen_deteccion, mascara_bgr, imagen_resultado]
    alto_img, ancho_img = paneles[0].shape[:2]

    lienzos = []
    for panel, etiqueta in zip(paneles, etiquetas):
        # Redimensionar al mismo alto/ancho por si difieren
        panel_r = cv2.resize(panel, (ancho_img, alto_img))

        # Barra de título negra con texto blanco
        barra = np.zeros((alto_etiqueta, ancho_img, 3), dtype=np.uint8)
        cv2.putText(
            barra, etiqueta,
            (8, alto_etiqueta - 8),
            FUENTE_CV2, 0.65, COLOR_TEXTO, 2, cv2.LINE_AA,
        )
        lienzos.append(np.vstack([barra, panel_r]))

    return np.hstack(lienzos)


def dibujar_mascaras(
    imagen: np.ndarray,
    mascaras: list[np.ndarray],
    ids_clase: list[int],
    alpha: float = ALPHA_MASCARA,
) -> np.ndarray:
    """
    Superpone máscaras de segmentación sobre una imagen NumPy BGR.

    Cada máscara se colorea según el índice de clase (ver COLOR_MASCARA).
    Se aplica como overlay semitransparente para no ocultar la imagen original.

    Parámetros
    ----------
    imagen     : Imagen NumPy BGR de fondo.
    mascaras   : Lista de arrays binarios (H, W) uint8, uno por detección.
                 Deben estar ya redimensionados al tamaño de la imagen.
    ids_clase  : Lista de índices de clase correspondientes a cada máscara.
    alpha      : Opacidad del overlay (0.0 = invisible, 1.0 = sólido).

    Retorna
    -------
    Imagen anotada con máscaras superpuestas (copia, no modifica la original).
    """
    if not mascaras:
        return imagen.copy()

    imagen_anotada = imagen.copy()
    overlay = imagen_anotada.copy()

    for mask, id_clase in zip(mascaras, ids_clase):
        color = COLOR_MASCARA.get(id_clase, (128, 128, 128))
        overlay[mask == 1] = color

    cv2.addWeighted(overlay, alpha, imagen_anotada, 1 - alpha, 0, imagen_anotada)
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
    Extrae cajas, scores, clases y máscaras del objeto Results de Ultralytics.

    Compatible con modelos de detección (solo cajas) y segmentación (cajas +
    máscaras). Si el modelo no produce máscaras, la clave 'mascaras' será una
    lista vacía.

    Parámetros
    ----------
    resultado : Objeto Results de Ultralytics (un solo frame).

    Retorna
    -------
    Diccionario con:
      - cajas_xyxy : list[list[float]] – coordenadas [x1, y1, x2, y2]
      - scores     : list[float]       – confianza de cada detección
      - clases     : list[str]         – nombre de clase de cada detección
      - ids_clase  : list[int]         – índice de clase de cada detección
      - mascaras   : list[np.ndarray]  – máscaras binarias (H, W) uint8,
                                         redimensionadas al tamaño original.
                                         Lista vacía si el modelo no es seg.
      - total      : int               – número total de detecciones
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

    # ── Máscaras de segmentación (solo disponibles con modelos *-seg) ──────
    mascaras = []
    if resultado.masks is not None:
        orig_h, orig_w = resultado.orig_shape
        for mask_tensor in resultado.masks.data:
            mask_np = mask_tensor.cpu().numpy()  # (H_modelo, W_modelo) float
            mask_resized = cv2.resize(
                mask_np,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )
            mascaras.append((mask_resized > 0.5).astype(np.uint8))

    return {
        "cajas_xyxy": cajas_xyxy,
        "scores": scores,
        "clases": clases,
        "ids_clase": ids_clase,
        "mascaras": mascaras,
        "total": len(cajas_xyxy),
    }

