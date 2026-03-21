"""
INFERENCIA – Taller YOLO Detección de Fachadas y Postes
=========================================================
API FastAPI para detección de fachadas y postes y eliminación de postes
usando YOLOv8-seg + LaMa (inpainting).

Uso:
    uvicorn src.inferencia:app --reload --port 8000
"""
import io
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from simple_lama_inpainting import SimpleLama

# Importar utilidades propias del proyecto
from src.utils import (
    armar_collage,
    bytes_a_numpy,
    dibujar_conteo_umbral,
    dibujar_detecciones,
    dibujar_mascaras,
    generar_mascara_postes,
    numpy_a_bytes,
    parsear_resultados_yolo,
)


# =====================================================
# CONFIGURACIÓN GLOBAL
# =====================================================

# Ruta por defecto al modelo entrenado (relativa a la raíz del proyecto)
RUTA_MODELO_DEFAULT = Path("models/postes-yolo.pt")

# Parámetros de inferencia
UMBRAL_CONFIANZA = 0.50
TAMANO_IMAGEN = 640

# Formatos de imagen aceptados
FORMATOS_PERMITIDOS = {"image/jpeg", "image/png", "image/bmp", "image/webp"}


# =====================================================
# INICIALIZACIÓN DE LA APLICACIÓN FASTAPI
# =====================================================

app = FastAPI(
    title="API – Detección de Fachadas y Postes con YOLO + LaMa",
    description=(
        "Detecta fachadas y postes en imágenes urbanas con YOLOv8-seg "
        "y elimina los postes usando inpainting con LaMa."
    ),
    version="2.0.0",
)


# =====================================================
# CARGA DE MODELOS 
# =====================================================

_modelo_cache: YOLO | None = None
_lama_cache: SimpleLama | None = None


def cargar_lama() -> SimpleLama:
    """Carga el modelo LaMa una sola vez y lo reutiliza en cada peticion."""
    global _lama_cache
    if _lama_cache is None:
        print("[INFO] Cargando modelo LaMa...")
        _lama_cache = SimpleLama()
        print("[INFO] LaMa cargado exitosamente")
    return _lama_cache


def cargar_modelo(ruta_pesos: str | Path = RUTA_MODELO_DEFAULT) -> YOLO:
    """
    Carga el modelo YOLO desde disco usando memoria cache.
    Solo se carga una vez durante el ciclo de vida de la API.

    Parametros
    ----------
    ruta_pesos : Ruta al archivo .pt de pesos entrenados.

    Retorna
    -------
    Instancia del modelo YOLO lista para inferencia.

    Lanza
    -----
    FileNotFoundError si el archivo de pesos no existe.
    """
    global _modelo_cache

    ruta_pesos = Path(ruta_pesos)

    # Verificar que el archivo de pesos exista antes de intentar cargarlo
    if not ruta_pesos.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de pesos en: {ruta_pesos}\n"
            f"Asegurate de haber entrenado el modelo (ver src/train_yolo.py) "
            f"y que los pesos estén en 'models/postes-yolo.pt'."
        )

    # Usar cache para no recargar el modelo en cada peticion
    if _modelo_cache is None:
        print(f"[INFO] Cargando modelo desde: {ruta_pesos}")
        _modelo_cache = YOLO(str(ruta_pesos))
        print("[INFO] Modelo cargado exitosamente")

    return _modelo_cache


def ejecutar_inferencia(imagen_bgr: np.ndarray) -> dict:
    """Ejecuta YOLO sobre una imagen NumPy BGR y retorna los resultados parseados."""
    modelo = cargar_modelo()
    resultados = modelo.predict(
        source=imagen_bgr,
        conf=UMBRAL_CONFIANZA,
        imgsz=TAMANO_IMAGEN,
        verbose=False,
    )
    return parsear_resultados_yolo(resultados[0])


# =====================================================
# FUNCIÓN AUXILIAR DE VALIDACIÓN
# =====================================================

def validar_archivo_imagen(archivo: UploadFile) -> bytes:
    """
    Valida que el archivo subido sea una imagen permitida y devuelve sus bytes.

    Parametros
    ----------
    archivo : Archivo subido via FastAPI UploadFile.

    Retorna
    -------
    Bytes del contenido del archivo.

    Lanza
    -----
    HTTPException 400 si el formato no es valido.
    """
    if archivo.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato no soportado: '{archivo.content_type}'. "
                f"Formatos validos: {sorted(FORMATOS_PERMITIDOS)}"
            ),
        )
    return archivo.file.read()


# =====================================================
# ENDPOINT DE LA API
# =====================================================

@app.get("/", summary="Informacion de la API")
def raiz():
    """Devuelve informacion general y lista de endpoints disponibles."""
    return {
        "api": "Detección de Fachadas y Postes usando YOLO y LaMa",
        "version": "2.0.0",
        "modelo": str(RUTA_MODELO_DEFAULT),
        "endpoints": {
            "POST /detectar_fachadas_postes": "Permite detectar fachadas y postes y devuelve una imagen con máscaras y bounding boxes.",
            "POST /borrar_postes": "Detecta fachadas y postes, eliminando estos ultimos con generación de máscaras y uso de Lama inpainting.",
        },
    }


@app.post("/detectar_fachadas_postes", summary="Detectar fachadas y postes")
async def detectar_fachadas_postes(
    archivo: UploadFile = File(description="Imagen JPG/PNG/BMP/WebP"),
):
    """
    Recibe una imagen y la devuelve con las detecciones dibujadas.

    La respuesta es una imagen JPEG que incluye:
    - Máscaras de segmentación coloreadas por clase (fachada / poste).
    - Bounding boxes con etiqueta de clase y confianza.
    - Contador total de detecciones.

    Headers de respuesta:
    - X-Umbral-Confianza
    - X-Detecciones-Total
    """
    # Validar y leer el archivo subido
    contenido = validar_archivo_imagen(archivo)

    # Convertir bytes a imagen NumPy BGR
    try:
        imagen_bgr = bytes_a_numpy(contenido)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Ejecutar inferencia
    try:
        datos = ejecutar_inferencia(imagen_bgr)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Dibujar mascaras de segmentacion
    imagen_anotada = dibujar_mascaras(
        imagen_bgr,
        datos["mascaras"],
        datos["ids_clase"],
    )

    # Dibujar bounding boxes y etiquetas sobre la imagen con mascaras
    imagen_anotada = dibujar_detecciones(
        imagen_anotada,
        datos["cajas_xyxy"],
        datos["scores"],
        datos["clases"],
        datos["ids_clase"],
        umbral_confianza=UMBRAL_CONFIANZA,
    )

    imagen_anotada = dibujar_conteo_umbral(imagen_anotada, datos["total"], UMBRAL_CONFIANZA)

    # Convertir imagen anotada a bytes JPEG para la respuesta
    imagen_bytes = numpy_a_bytes(imagen_anotada, extension=".jpg")

    return StreamingResponse(
        io.BytesIO(imagen_bytes),
        media_type="image/jpeg",
        headers={
            "X-Umbral-Confianza": str(UMBRAL_CONFIANZA),
            "X-Detecciones-Total": str(datos["total"]),
            "Content-Disposition": f'inline; filename="deteccion_{archivo.filename}"',
        },
    )


@app.post("/borrar_postes", summary="Detecta fachadas y postes, eliminando estos ultimos con generación de máscaras y uso de Lama inpainting.")
async def borrar_postes(
    archivo: UploadFile = File(description="Imagen JPG/PNG/BMP/WebP"),
):
    """
    Ejecuta el pipeline completo y devuelve un collage JPEG con tres paneles:

    | Deteccion (mascaras + bboxes) | Mascara postes | Sin postes (LaMa) |

    Útil para verificar visualmente cada etapa del proceso en una sola imagen.
    """
    contenido = validar_archivo_imagen(archivo)

    try:
        imagen_bgr = bytes_a_numpy(contenido)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        datos = ejecutar_inferencia(imagen_bgr)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    alto, ancho = imagen_bgr.shape[:2]

    # Panel 1 — deteccion con mascaras y bounding boxes
    imagen_deteccion = dibujar_mascaras(imagen_bgr, datos["mascaras"], datos["ids_clase"])
    imagen_deteccion = dibujar_detecciones(
        imagen_deteccion, datos["cajas_xyxy"], datos["scores"], datos["clases"], datos["ids_clase"],
        umbral_confianza=UMBRAL_CONFIANZA,
    )
    imagen_deteccion = dibujar_conteo_umbral(imagen_deteccion, datos["total"], UMBRAL_CONFIANZA)

    # Panel 2 — mascara binaria de postes (con dilatacion)
    mascara = generar_mascara_postes(datos["mascaras"], datos["ids_clase"], (alto, ancho))

    # Panel 3 — inpainting con LaMa
    if mascara.max() == 0:
        imagen_resultado = imagen_bgr.copy()
    else:
        imagen_rgb = PILImage.fromarray(imagen_bgr[:, :, ::-1])
        mascara_pil = PILImage.fromarray(mascara)
        lama = cargar_lama()
        resultado_pil = lama(imagen_rgb, mascara_pil)
        imagen_resultado = np.array(resultado_pil)[:, :, ::-1]

    collage = armar_collage(imagen_deteccion, mascara, imagen_resultado)
    collage_bytes = numpy_a_bytes(collage, extension=".jpg")

    n_postes = sum(1 for c in datos["ids_clase"] if c == 1)

    return StreamingResponse(
        io.BytesIO(collage_bytes),
        media_type="image/jpeg",
        headers={
            "X-Postes-Detectados": str(n_postes),
            "Content-Disposition": f'inline; filename="pipeline_{archivo.filename}"',
        },
    )
