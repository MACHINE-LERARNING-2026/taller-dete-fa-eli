"""
INFERENCIA – Taller YOLO Detección de Casas
============================================
API FastAPI + script standalone para ejecutar detección de casas
usando el modelo entrenado 'models/postes-yolo.pt'.

Uso como API:
    uvicorn src.inferencia:app --reload --port 8000

Uso como script (CLI):
    python src/inferencia.py --imagen ruta/imagen.jpg
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
    title="API – Detector de Casas con YOLO",
    description=(
        "Detecta casas en imágenes usando un modelo YOLOv8 entrenado "
        "con imágenes colombianas urbanas y rurales."
    ),
    version="1.2.0",
)


# =====================================================
# CARGA DE MODELOS (singleton para no recargar en cada request)
# =====================================================

_modelo_cache: YOLO | None = None
_lama_cache: SimpleLama | None = None


def cargar_lama() -> SimpleLama:
    """Carga el modelo LaMa una sola vez y lo reutiliza en cada petición."""
    global _lama_cache
    if _lama_cache is None:
        print("[INFO] Cargando modelo LaMa...")
        _lama_cache = SimpleLama()
        print("[INFO] LaMa cargado exitosamente ✅")
    return _lama_cache


def cargar_modelo(ruta_pesos: str | Path = RUTA_MODELO_DEFAULT) -> YOLO:
    """
    Carga el modelo YOLO desde disco usando caché en memoria.
    Solo se carga una vez durante el ciclo de vida de la API.

    Parámetros
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
            f"Asegúrate de haber entrenado el modelo (ver src/train_yolo.py) "
            f"y que los pesos estén en 'models/postes-yolo.pt'."
        )

    # Usar caché para no recargar el modelo en cada petición
    if _modelo_cache is None:
        print(f"[INFO] Cargando modelo desde: {ruta_pesos}")
        _modelo_cache = YOLO(str(ruta_pesos))
        print("[INFO] Modelo cargado exitosamente ✅")

    return _modelo_cache


def ejecutar_inferencia(imagen_bgr) -> tuple[dict, object]:
    """
    Ejecuta la inferencia YOLO sobre una imagen NumPy BGR.

    Parámetros
    ----------
    imagen_bgr : Imagen en formato NumPy BGR.

    Retorna
    -------
    Tupla (resultados_parseados, objeto_resultado_ultralytics).
    """
    modelo = cargar_modelo()

    # Ejecutar predicción
    resultados = modelo.predict(
        source=imagen_bgr,
        conf=UMBRAL_CONFIANZA,
        imgsz=TAMANO_IMAGEN,
        verbose=False,
    )

    # Tomamos el primer (y único) frame de resultados
    resultado_frame = resultados[0]
    datos = parsear_resultados_yolo(resultado_frame)

    return datos, resultado_frame


# =====================================================
# FUNCIÓN AUXILIAR DE VALIDACIÓN
# =====================================================

def validar_archivo_imagen(archivo: UploadFile) -> bytes:
    """
    Valida que el archivo subido sea una imagen permitida y devuelve sus bytes.

    Parámetros
    ----------
    archivo : Archivo subido vía FastAPI UploadFile.

    Retorna
    -------
    Bytes del contenido del archivo.

    Lanza
    -----
    HTTPException 400 si el formato no es válido.
    """
    if archivo.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato no soportado: '{archivo.content_type}'. "
                f"Formatos válidos: {sorted(FORMATOS_PERMITIDOS)}"
            ),
        )
    return archivo.file.read()


# =====================================================
# ENDPOINT DE LA API
# =====================================================

@app.get("/", summary="Información de la API")
def raiz():
    """Devuelve información general y lista de endpoints disponibles."""
    return {
        "api": "Detector de Casas – YOLO",
        "version": "1.2.0",
        "modelo": str(RUTA_MODELO_DEFAULT),
        "endpoints": {
            "POST /detectar_casas": "Detectar casas y devolver imagen con las detecciones dibujadas.",
        },
    }


@app.post("/detectar_casas", summary="Detectar casas")
async def detectar_casas(
    archivo: UploadFile = File(description="Imagen JPG/PNG/BMP/WebP"),
):
    """
    Recibe una imagen y devuelve la imagen con las detecciones dibujadas.

    La respuesta es una imagen JPEG que incluye:
    - Bounding boxes en formato [x1, y1, x2, y2] dibujadas sobre la imagen.
    - Etiquetas con nombre de clase y score de confianza.
    - Contador total de casas detectadas.
    - Umbral de confianza utilizado.

    Información adicional se envía en los headers HTTP:
    - X-Umbral-Confianza
    - X-Casas-Detectadas
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
        datos, _ = ejecutar_inferencia(imagen_bgr)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Dibujar máscaras de segmentación (si el modelo las produce)
    imagen_anotada = dibujar_mascaras(
        imagen_bgr,
        datos["mascaras"],
        datos["ids_clase"],
    )

    # Dibujar bounding boxes y etiquetas sobre la imagen con máscaras
    imagen_anotada = dibujar_detecciones(
        imagen_anotada,
        datos["cajas_xyxy"],
        datos["scores"],
        datos["clases"],
        umbral_confianza=UMBRAL_CONFIANZA,
    )

    # Añadir contador de casas
    imagen_anotada = dibujar_conteo_umbral(imagen_anotada, datos["total"], UMBRAL_CONFIANZA)

    # Convertir imagen anotada a bytes JPEG para la respuesta
    imagen_bytes = numpy_a_bytes(imagen_anotada, extension=".jpg")

    return StreamingResponse(
        io.BytesIO(imagen_bytes),
        media_type="image/jpeg",
        headers={
            "X-Umbral-Confianza": str(UMBRAL_CONFIANZA),
            "X-Casas-Detectadas": str(datos["total"]),
            "Content-Disposition": f'inline; filename="deteccion_{archivo.filename}"',
        },
    )


@app.post("/mascara_postes", summary="Obtener máscara binaria de postes")
async def mascara_postes(
    archivo: UploadFile = File(description="Imagen JPG/PNG/BMP/WebP"),
):
    """
    Devuelve la máscara binaria de los postes detectados en la imagen.

    La respuesta es una imagen PNG en escala de grises donde:
    - Blanco (255) = región de poste a eliminar.
    - Negro  (0)   = fondo a conservar.

    Esta máscara puede usarse directamente como entrada para LaMa u otro
    modelo de inpainting.
    """
    contenido = validar_archivo_imagen(archivo)

    try:
        imagen_bgr = bytes_a_numpy(contenido)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        datos, _ = ejecutar_inferencia(imagen_bgr)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    alto, ancho = imagen_bgr.shape[:2]
    mascara = generar_mascara_postes(datos["mascaras"], datos["ids_clase"], (alto, ancho))

    mascara_bytes = numpy_a_bytes(mascara, extension=".png")

    return StreamingResponse(
        io.BytesIO(mascara_bytes),
        media_type="image/png",
        headers={
            "X-Postes-Detectados": str(sum(1 for c in datos["ids_clase"] if c == 1)),
            "Content-Disposition": f'inline; filename="mascara_{archivo.filename}"',
        },
    )


@app.post("/eliminar_postes", summary="Eliminar postes con inpainting (LaMa)")
async def eliminar_postes(
    archivo: UploadFile = File(description="Imagen JPG/PNG/BMP/WebP"),
):
    """
    Pipeline completo: detecta postes, genera su máscara y aplica LaMa
    para reconstruir la imagen sin los postes.

    Pasos internos:
    1. YOLO detecta postes y genera máscaras de segmentación.
    2. Las máscaras de postes se combinan en una imagen binaria.
    3. LaMa reconstruye la región enmascarada con contenido coherente.

    La respuesta es la imagen original sin los postes, en formato JPEG.
    """
    contenido = validar_archivo_imagen(archivo)

    try:
        imagen_bgr = bytes_a_numpy(contenido)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        datos, _ = ejecutar_inferencia(imagen_bgr)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    alto, ancho = imagen_bgr.shape[:2]
    mascara = generar_mascara_postes(datos["mascaras"], datos["ids_clase"], (alto, ancho))

    if mascara.max() == 0:
        # No hay postes detectados — devolver imagen original sin cambios
        imagen_bytes = numpy_a_bytes(imagen_bgr, extension=".jpg")
        return StreamingResponse(
            io.BytesIO(imagen_bytes),
            media_type="image/jpeg",
            headers={
                "X-Postes-Eliminados": "0",
                "Content-Disposition": f'inline; filename="sin_postes_{archivo.filename}"',
            },
        )

    # Convertir imagen BGR (OpenCV) a RGB (PIL) para LaMa
    imagen_rgb = PILImage.fromarray(imagen_bgr[:, :, ::-1])
    mascara_pil = PILImage.fromarray(mascara)

    lama = cargar_lama()
    imagen_resultado = lama(imagen_rgb, mascara_pil)  # Retorna PIL RGB

    # Convertir resultado PIL RGB → NumPy BGR → bytes
    resultado_bgr = np.array(imagen_resultado)[:, :, ::-1]
    imagen_bytes = numpy_a_bytes(resultado_bgr, extension=".jpg")

    n_postes = sum(1 for c in datos["ids_clase"] if c == 1)

    return StreamingResponse(
        io.BytesIO(imagen_bytes),
        media_type="image/jpeg",
        headers={
            "X-Postes-Eliminados": str(n_postes),
            "Content-Disposition": f'inline; filename="sin_postes_{archivo.filename}"',
        },
    )


@app.post("/pipeline_completo", summary="Collage: detección + máscara + resultado")
async def pipeline_completo(
    archivo: UploadFile = File(description="Imagen JPG/PNG/BMP/WebP"),
):
    """
    Ejecuta el pipeline completo y devuelve un collage JPEG con tres paneles:

    | Detección (máscaras + bboxes) | Máscara postes | Sin postes (LaMa) |

    Útil para verificar visualmente cada etapa del proceso en una sola imagen.
    """
    contenido = validar_archivo_imagen(archivo)

    try:
        imagen_bgr = bytes_a_numpy(contenido)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        datos, _ = ejecutar_inferencia(imagen_bgr)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    alto, ancho = imagen_bgr.shape[:2]

    # Panel 1 — detección con máscaras y bounding boxes
    imagen_deteccion = dibujar_mascaras(imagen_bgr, datos["mascaras"], datos["ids_clase"])
    imagen_deteccion = dibujar_detecciones(
        imagen_deteccion, datos["cajas_xyxy"], datos["scores"], datos["clases"],
        umbral_confianza=UMBRAL_CONFIANZA,
    )
    imagen_deteccion = dibujar_conteo_umbral(imagen_deteccion, datos["total"], UMBRAL_CONFIANZA)

    # Panel 2 — máscara binaria de postes (con dilatación)
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
