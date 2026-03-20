--------------------------------Detector de Casas (YOLO)------------------------------
-------------------------------------------------------------------------------------
Contenido:

1. Descripción
2. Estructura del Proyecto
3. Requisitos
4. Descarga del repositorio
	4.1. Clonar/Descargar el repositorio
	4.2. Descargar como .zip
5. Ejecución
	5.1. Crear y activar un entorno virtual
	5.2. Entrenamiento (opcional)
	5.3. Iniciar servicios de FastAPI
6. Resultados (métricas) y ejemplos
7. Limitaciones y pasos futuros recomendados
--------------------------------------------------------------------------------------
1. Descripción
Este proyecto implementa un API para detección de casas usando YOLO (Ultralytics YOLOv8) con arquitectura modular en Python.

El sistema permite:
- Entrenar un modelo de detección de casas sobre un dataset personalizado y parametrizado con Roboflow. 
- Ejecutar un analisis de imagenes con la finalidad de detectar casas mediante un servicio HTTP (FastAPI).

Formatos de imagen soportados:
- .png, .jpeg, .webp, .bmp

Nota: Los comandos en este README se muestran con `python` / `pip`. Si en tu entorno el intérprete es `python3` / `pip3`, sustitúyelos según corresponda.
-------------------------------------------------------------------------------------
2. Estructura del Proyecto

```
taller-yolo-casas/
├── data.yaml               # Descriptor del dataset (rutas train/val/test, nc, names)
├── src/
│   ├── train_yolo.py       # Script de entrenamiento
│   ├── inferencia.py       # Servicio FastAPI para inferencia
│   └── utils.py            # Funciones utilitarias (Dibujar Conteo Umbral, Dibuja bounding boxes, parseo de resultados, etc.)
├── models/
│   └── detect/             # Carpeta que genera Ultralytics durante el training y contiene la matriz de confusion y resultados de entrenamiento
│   └── weights/		    # contiene los pesos preentrenados de yolov8m
│   └── house-yolo.pt		# Resultado de pesos del entrenamiento que seran usados por la API
├── train/                  # Imágenes y etiquetas de entrenamiento generadas por Roboflow
├── valid/                  # Imágenes y etiquetas de validación generadas por Roboflow
├── requirements.txt		# Dependencias y librerias necesarias para la aplicación
└── README.md				# Descripción del repositorio
```

Descripción del dataset y origen de imágenes
- Archivo descriptor: `data.yaml` (contiene rutas a `train`, `val`, `test` y el número de clases `nc`).
- Clases: `['Casa']` (1 clase) — confirma en `data.yaml`.
- Origen: dataset exportado desde Roboflow (ver `data.yaml` para detalles del workspace/proyecto/versión).
- Estructura esperada de carpetas (relativa a la raíz del repo):
	- `train/images`, `train/labels`
	- `valid/images`, `valid/labels`
	- `test/images`, `test/labels`

-------------------------------------------------------------------------------------
3. Requisitos
Para el correcto funcionamiento del proyecto es necesario:
- Python 3.9 o superior
- `pip` actualizado
- GPU opcional (para entrenar más rápido)

para validar la versión de Python usa el siguiente comando:

```bash
python --version
```
Nota: Para GPU, sigue las instrucciones de la web oficial para instalar la build de `torch` compatible con tu versión de CUDA.
-------------------------------------------------------------------------------------
4. Descarga del repositorio

4.1 Clonar/Descargar el repositorio

```bash
git clone https://github.com/MACHINE-LERARNING-2026/taller-yolo-casas.git
cd taller-yolo-casas
```

4.2 Descargar como .zip

Desde la interfaz web del repositorio https://github.com/MACHINE-LERARNING-2026/taller-yolo-casas.git descarga el ZIP, descomprímelo y entra en la carpeta:

```bash
cd taller-yolo-casas
```
-------------------------------------------------------------------------------------
5. Ejecución

5.1. Crear y activar un entorno virtual

Para windows se utiliza el siguiente comando desde un CMD y ubicarse sobre la ruta del repositorio (taller-yolo-casas):
```bash
# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Para macOS o Linux se utiliza el siguiente comando desde la terminal y ubicarse sobre la ruta del repositorio (taller-yolo-casas):
```bash
# macOS / Linux
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Si usas macOS y encuentras problemas SSL con algunas librerías, instala los certificados del sistema Python si aplica:

```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```
Nota: Si `uvicorn` o instalación fallan revisa que el `venv` esté activado.

5.2. Entrenamiento (opcional)
El repositorio ya cuenta los pesos entrenados y estan ubicados en: /models/house-yolo.pt por lo cual no es necesario realizar nuevamente un entrenamiento para usar la API, por lo cual este proceso de entrenamiento es totalmente opcional.

```bash
# Ejecuta el script de entrenamiento
python src/train_yolo.py
```

El script guarda los resultados en `models/detect/weights/best.pt` y copia `best.pt` a `models/house-yolo.pt`.

5.3. Iniciar servicios de FastAPI

desde un cmd de windows y ubicados sobre la raiz del repositorio se debe ejecutar el siguiente comando:

```bash
uvicorn src.inferencia:app --reload
```
posterior a la confirmación de la ejecución se debe abrir la URL `http://127.0.0.1:8000/docs` desde tu navegador de preferencia. Esta URL tiene los siguientes metodos permitidos de la API:
- GET / 					Información de la API
- POST /detectar_casas 		Recibe una imagen y devuelve otra imagen con las detecciones de casas realizadas

ejemplo de POST:
- Despliega el metodo POST /detectar_casas y dar click en Try it out, posteriormente oprimir el boton de seleccionar archivo y seleccionar la imagen a la que se le quiere realizar la detección. El resultado puede ser similar a este:
<img width="1692" height="953" alt="Casa93" src="https://github.com/user-attachments/assets/29247730-3282-4567-bce7-51a35f1fed37" />

-------------------------------------------------------------------------------------
6. Resultados (métricas) y ejemplos de detección

Para validar y obtener métricas con Ultralytics:

```python
from ultralytics import YOLO
model = YOLO('models/house-yolo.pt')
metrics = model.val(data='data.yaml')
print(metrics)
```

Con el dataset de 139 imagenes divididas en 111 para entramiento y 28 de validación se obtuvieron las siguientes metricas después del entrenamiento:

<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/0cc71254-0eef-44a8-abdf-526e7c6141d8" />

Los resultados del entrenamiento muestran una disminución progresiva en las funciones de pérdida tanto en entrenamiento como en validación, lo que indica que el modelo está aprendiendo adecuadamente a localizar y clasificar mejor las casas presentes en las imágenes.

Las métricas de evaluación presentan una tendencia creciente a lo largo de las épocas, alcanzando aproximadamente los siguientes valores:
- mAP@0.5 (Mean Average Precision): 0.5635912229825473
Esta métrica mide el rendimiento global del modelo considerando precisión y recall al mismo tiempo.
- Precision: 0.6734037212645044
Aproximadamente 67 de cada 100 detecciones realizadas por el modelo sí corresponden a casas reales. El restante 33% corresponde a falsas detecciones, es decir, el modelo identifica como casa algo que en realidad no lo es.
- Recall: 0.47096774193548385
El modelo detecta aproximadamente el 47% de todas las casas reales presentes en las imágenes.

Esto evidencia una mejora gradual en la capacidad del modelo para detectar casas, aunque todavía existen casos en los que algunos casas reales no son detectadas.

En general, el modelo muestra un comportamiento de aprendizaje estable y sin señales claras de sobreajuste, ya que las pérdidas de validación también disminuyen durante el entrenamiento. No obstante, los resultados sugieren que el desempeño podría mejorarse mediante el uso de más datos de entrenamiento, optimización de anotaciones o ajuste de hiperparámetros en el modelo basado en YOLOv8.

A continuaación, se observa la matriz de confusión obtenida:

<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/5a2ec5d8-4843-4993-a6d6-b582ca1072a8" />

En esta matriz se puede apreciar que el modelo logra identificar correctamente 93 instancias de casas, lo que evidencia una capacidad adecuada para reconocer la clase objetivo. Sin embargo, también se observa un número considerable de falsos positivos (94 casos), donde el modelo predice la presencia de una casa cuando en realidad corresponde al fondo de la imagen, lo que indica cierta confusión entre estructuras del entorno y la clase de interés. Asimismo, se registran 62 falsos negativos, es decir, casos en los que el modelo no logra detectar una casa presente y la clasifica como background. En conjunto, estos resultados sugieren que, aunque el modelo presenta un desempeño razonable en la detección de casas, todavía existe margen de mejora en la discriminación entre la clase objetivo y el fondo.

Un ejemplo de detección de falso positivo es el siguiente:

![ac77c22d-0583-4f4b-a6d1-f030c5f29947](https://github.com/user-attachments/assets/90b5f884-a7ac-46b0-beec-572a2a857617)

-------------------------------------------------------------------------------------
7. Limitaciones y pasos futuros recomendados
- Si el dataset es pequeño existe riesgo de sobreajuste. Recomendaciones:
	- Aumentar datos (augmentations): rotaciones, flips, variaciones de brillo/contraste.
	- Recolectar más imágenes en distintas condiciones (iluminación, ángulos, entornos).
	- Realizar validación cruzada o usar técnicas de regularización.
	- Evaluar mAP en múltiples umbrales (mAP@[.5:.95]).


- Despliegue:
	- En producción, ejecutar Uvicorn sin `--reload` y orquestar con systemd/Docker/Gunicorn.
	- Añadir tests automáticos y CI que verifique endpoints y una inferencia mínima.

---

Fecha de actualización: Marzo 2026





