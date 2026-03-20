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
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO

# Importar utilidades propias del proyecto
from src.utils import (
    bytes_a_numpy,
    dibujar_conteo_umbral,
    dibujar_detecciones,
    numpy_a_bytes,
    parsear_resultados_yolo,
)


# =====================================================
# CONFIGURACIÓN GLOBAL
# =====================================================

# Ruta por defecto al modelo entrenado (relativa a la raíz del proyecto)
RUTA_MODELO_DEFAULT = Path("models/postes-yolo.pt")

# Parámetros de inferencia
UMBRAL_CONFIANZA = 0.70
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
# CARGA DEL MODELO (singleton para no recargar en cada request)
# =====================================================

_modelo_cache: YOLO | None = None


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
        "version": "1.0.0",
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

    # Dibujar detecciones sobre la imagen original
    imagen_anotada = dibujar_detecciones(
        imagen_bgr,
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
