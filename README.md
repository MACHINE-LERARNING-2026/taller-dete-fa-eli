----------Detector de Fachadas y Postes (YOLO)----------

--------------Eliminación de Postes (LaMa)--------------

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

--------------------------------------------------------

1. Descripción
Este proyecto implementa una API para detección de fachadas y postes usando YOLO (Ultralytics YOLOv8) con arquitectura modular en Python. Adicionalmente, integra LaMa (Large Mask inpainting) para la eliminación automática de postes detectados en las imágenes.

El sistema permite:

Entrenar un modelo de detección de fachadas y postes sobre un dataset personalizado y parametrizado con Roboflow.
Ejecutar un análisis de imágenes con la finalidad de detectar fachadas y postes mediante un servicio HTTP (FastAPI).
Eliminar automáticamente los postes detectados en una imagen utilizando LaMa para reconstruir el fondo de forma coherente.

Formatos de imagen soportados:

.png, .jpeg, .webp, .bmp

Nota: Los comandos en este README se muestran con python / pip. Si en tu entorno el intérprete es python3 / pip3, sustitúyelos según corresponda.

2. Estructura del Proyecto

taller-yolo-fachadas/
├── data.yaml               # Descriptor del dataset (rutas train/val/test, nc, names)
├── src/
│   ├── train_yolo.py       # Script de entrenamiento
│   ├── inferencia.py       # Servicio FastAPI para inferencia y eliminación de postes
│   └── utils.py            # Funciones utilitarias (Dibujar Conteo Umbral, Dibuja bounding boxes,
│                           # parseo de resultados, enmascaramiento de postes, integración con LaMa, etc.)
├── models/
│   └── detect/             # Carpeta que genera Ultralytics durante el training y contiene la matriz de confusión y resultados de entrenamiento
│   └── weights/            # Contiene los pesos preentrenados de yolov8m
│   └── fachadas-yolo.pt    # Resultado de pesos del entrenamiento que serán usados por la API
├── train/                  # Imágenes y etiquetas de entrenamiento generadas por Roboflow
├── valid/                  # Imágenes y etiquetas de validación generadas por Roboflow
├── requirements.txt        # Dependencias y librerías necesarias para la aplicación
└── README.md               # Descripción del repositorio

Descripción del dataset y origen de imágenes

Archivo descriptor: data.yaml (contiene rutas a train, val, test y el número de clases nc).
Clases: ['Fachada', 'Poste'] (2 clases)
Origen: dataset exportado desde Roboflow
Estructura esperada de carpetas (relativa a la raíz del repo):

train/images, train/labels
valid/images, valid/labels
test/images, test/labels

3. Requisitos
Para el correcto funcionamiento del proyecto es necesario:


Python 3.9 o superior
pip actualizado
GPU opcional (para entrenar más rápido y acelerar la inferencia de LaMa)

Para validar la versión de Python usa el siguiente comando:
bashpython --version
Nota: Para GPU, sigue las instrucciones de la web oficial para instalar la build de torch compatible con tu versión de CUDA.

4. Descarga del repositorio

4.1 Clonar/Descargar el repositorio
bashgit clone https://github.com/MACHINE-LERARNING-2026/taller-dete-fa-eli.git
cd taller-dete-fa-eli
4.2 Descargar como .zip
Desde la interfaz web del repositorio https://github.com/MACHINE-LERARNING-2026/taller-dete-fa-eli.git descarga el ZIP, descomprímelo y entra en la carpeta:
bashcd taller-dete-fa-eli

5. Ejecución

5.1. Crear y activar un entorno virtual

Para windows se utiliza el siguiente comando desde un CMD y ubicarse sobre la ruta del repositorio (taller-dete-fa-eli):
bash# Windows

  python -m venv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  
Para macOS o Linux se utiliza el siguiente comando desde la terminal y ubicarse sobre la ruta del repositorio (taller-dete-fa-eli.git):

  bash# macOS / Linux
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  
Si usas macOS y encuentras problemas SSL con algunas librerías, instala los certificados del sistema Python si aplica:

  bash/Applications/Python\ 3.x/Install\ Certificates.command
  
Nota: Si uvicorn o la instalación fallan, revisa que el venv esté activado.

5.2. Entrenamiento (opcional)

El repositorio ya cuenta con los pesos entrenados ubicados en: /models/postes-yolo.pt, por lo cual no es necesario realizar nuevamente un entrenamiento para usar la API. Este proceso es totalmente opcional.

  bash# Ejecuta el script de entrenamiento
  python src/train_yolo.py

El script guarda los resultados en models/detect/weights/best.pt y copia best.pt a models/postes-yolo.pt

5.3. Iniciar servicios de FastAPI

Desde un CMD de Windows y ubicados sobre la raíz del repositorio, ejecuta el siguiente comando:

  bashuvicorn src.inferencia:app --reload

Posterior a la confirmación de la ejecución, abre la URL http://127.0.0.1:8000/docs desde tu navegador de preferencia. Esta URL expone los siguientes métodos de la API:

GET /                         Información de la API

POST /detectar_fachadas_postes  Recibe una imagen y devuelve otra imagen con las detecciones de fachadas y postes realizadas

POST /borrar_postes           Recibe una imagen, detecta los postes, los enmascara y aplica LaMa para eliminarlos, devolviendo la imagen final sin postes

Ejemplo de POST /detectar_fachadas_postes:

Despliega el método POST /detectar_fachadas_postes y da click en Try it out, posteriormente oprime el botón de seleccionar archivo y selecciona la imagen a la que se le quiere realizar la detección. La respuesta es la imagen original con los bounding boxes de fachadas y postes dibujados.

Ejemplo de POST /borrar_postes:

Despliega el método POST /borrar_postes y da click en Try it out, selecciona el archivo de imagen. La API realizará internamente los siguientes pasos:

- Detección de fachadas y postes con YOLO.
- Generación de una máscara binaria sobre las regiones de los postes detectados.
- Procesamiento con LaMa para reconstruir el fondo en las zonas enmascaradas.
- Retorno de la imagen resultante sin los postes.

6. Resultados (métricas) y ejemplos de detección

Para validar y obtener métricas con Ultralytics:

  pythonfrom ultralytics import YOLO
  model = YOLO('models/postes-yolo.pt')
  metrics = model.val(data='data.yaml')
  print(metrics)
  
Con el dataset de 426 imágenes divididas en 352 para entrenamiento, 37 de validación y 37 de test. Se obtuvieron las siguientes métricas después del entrenamiento:

<img width="4000" height="1200" alt="results" src="https://github.com/user-attachments/assets/bff4970c-c10b-458b-baf9-3427be78fa28" />

Los resultados del entrenamiento muestran una disminución progresiva en las funciones de pérdida tanto en entrenamiento como en validación, lo que indica que el modelo está aprendiendo adecuadamente a localizar y clasificar las fachadas y postes presentes en las imágenes.

Las métricas de evaluación presentan una tendencia creciente a lo largo de las épocas, alcanzando aproximadamente los siguientes valores:

- mAP@0.5 (Mean Average Precision): 0.5635912229825473 Esta métrica mide el rendimiento global del modelo considerando precisión y recall al mismo tiempo.
- Precision: 0.6734037212645044. Aproximadamente 67 de cada 100 detecciones realizadas por el modelo corresponden a fachadas o postes reales. El restante 33% corresponde a falsas detecciones.
- Recall: 0.47096774193548385. El modelo detecta aproximadamente el 47% de todas las fachadas y postes reales presentes en las imágenes.

Esto evidencia una mejora gradual en la capacidad del modelo para detectar los objetos de interés, aunque todavía existen casos en los que algunas instancias reales no son detectadas.

En general, el modelo muestra un comportamiento de aprendizaje estable y sin señales claras de sobreajuste, ya que las pérdidas de validación también disminuyen durante el entrenamiento. No obstante, los resultados sugieren que el desempeño podría mejorarse mediante el uso de más datos de entrenamiento, optimización de anotaciones o ajuste de hiperparámetros en el modelo basado en YOLOv8.
A continuación, se observa la matriz de confusión obtenida:

<img width="3000" height="2250" alt="confusion_matrix" src="https://github.com/user-attachments/assets/5a2ec5d8-4843-4993-a6d6-b582ca1072a8" />

En esta matriz se puede apreciar que el modelo logra identificar correctamente un número considerable de instancias de fachadas y postes. Sin embargo, también se observan falsos positivos, donde el modelo predice la presencia de una clase cuando en realidad corresponde al fondo de la imagen, lo que indica cierta confusión entre estructuras del entorno y las clases de interés. Asimismo, se registran falsos negativos, es decir, casos en los que el modelo no logra detectar un objeto presente y lo clasifica como background. En conjunto, estos resultados sugieren que, aunque el modelo presenta un desempeño razonable, todavía existe margen de mejora en la discriminación entre las clases objetivo y el fondo.

7. Limitaciones y pasos futuros recomendados

Si el dataset es pequeño existe riesgo de sobreajuste. Recomendaciones:

- Aumentar datos (augmentations): rotaciones, flips, variaciones de brillo/contraste.
- Recolectar más imágenes en distintas condiciones (iluminación, ángulos, entornos).
- Realizar validación cruzada o usar técnicas de regularización.
- Evaluar mAP en múltiples umbrales (mAP@[.5:.95]).

Calidad del inpainting con LaMa:

- El resultado de la eliminación de postes depende de la calidad de la máscara generada. Máscaras imprecisas o muy grandes pueden producir artefactos visuales.
- En escenas con fondos complejos (vegetación densa, edificios con texturas variadas), LaMa puede generar reconstrucciones menos realistas.
- Se recomienda refinar las máscaras de postes con técnicas de dilatación/erosión para mejorar los resultados del inpainting.

----------------------------------------

Fecha de actualización: Marzo 2026





