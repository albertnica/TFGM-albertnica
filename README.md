# Predicción del Redshift mediante Modelos de Machine Learning

Este repositorio contiene varios modelos de machine learning que se comparan entre sí con el objetivo de predecir el redshift a partir de espectros astronómicos obtenidos del Sloan Digital Sky Survey (SDSS).

## Descripción

El proyecto implementa y evalúa diferentes modelos de machine learning para la predicción del redshift a partir de datos longitud de onda - flujo (habiendo eliminado el ruido lumínico ambiental previamente). Los modelos incluidos en este proyecto son:

- **MLP (Multi-Layer Perceptron):** Redes neuronales tradicionales que ayudan a modelar relaciones complejas en los datos.
- **CNN (Convolutional Neural Network):** Redes neuronales convolucionales, que se utilizan para extraer características relevantes de los espectros.

## Estructura del Proyecto

- **/data:** Carpeta con los datasets utilizados durante el entrenamiento de los modelos.
- **[/extra](extra):** Carpeta auxiliar usada para la obtención y procesamiento de los datos y para la construcción de los datasets de entrenamiento.
- **[/SciServer](SciServer):** Carpeta con todos los archivos necesarios para que funcionen los comandos del módulo SciServer, para evitar instalaciones del mismo.
- **/spectrums:** Carpeta con varios espectros descargados con [Fits Retriever](<Fits Retriever.ipynb>) que pueden usarse para probar los modelos (última línea del notebook de cualquier modelo).
- **/storage:** Carpeta destinada a almacenar los modelos entrenados y datos de evaluaciones.
- **[1 Fits Retriever](<1 Fits Retriever.ipynb>):** Extrae todos los archivos .fits de la base de datos de la SDSS y los descarga.
- **[2 Spaectra Processor](<2 Spectra Processor.ipynb>):** Programa clave para la obtención del dataset principal de entrenamiento, creado para seleccionar de manera un poco más uniforme los .fits sobre todo para que se aprovechen mucho más los datos con Z ≥ 4.
- **[3 Dataset Builder](<3 Dataset Builder.ipynb>):** Toma todos los .fits de una carpeta y construye el dataset de entrenamiento, que consta únicamente de los datos de fluxes/wavelenghts y de Z, y los guarda en la carpeta /data de [Onedrive](https://unioviedo-my.sharepoint.com/:f:/g/personal/uo284776_uniovi_es/EiifkyAiWZ9HqcUwLZppl-4BypXdYoWkSBb3XhzWpCAhqw?e=Zl62pY). Es el archivo estándar que marca como se deben procesar los archivos antes de pasarselos a los modelos, ya que el input debe ser obligatoriamente de 5000 fluxes + 5000 wavelenghts.

<br />

- **[4.1 Model MLP 600k](<4.1 Model MLP 600k.ipynb>), [4.1 Model MLP 1M](<4.1 Model MLP 1M.ipynb>):** Modelo basado en Perceptrones Multicapa.

<br />

- **[4.2 Model CNN 600k 0](<4.2 Model CNN 600k 0.ipynb>), [4.2 Model CNN 1M 0](<4.2 Model CNN 1M 0.ipynb>):** Modelo básico sustentado en redes convolucionales.
- **[4.2 Model CNN 600k 1](<4.2 Model CNN 600k 1.ipynb>), [4.2 Model CNN 1M 1](<4.2 Model CNN 1M 1.ipynb>):** Se introduce el concepto de dropout en la capa lineal.
- **[4.2 Model CNN 600k 2](<4.2 Model CNN 600k 2.ipynb>), [4.2 Model CNN 1M 2](<4.2 Model CNN 1M 2.ipynb>):** Se amplía el número de capas convolucionales.
- **[4.2 Model CNN 600k 3](<4.2 Model CNN 600k 3.ipynb>), [4.2 Model CNN 1M 3](<4.2 Model CNN 1M 3.ipynb>):** Se introduce un módulo de atención.
- **[4.2 Model CNN 600k 4](<4.2 Model CNN 600k 4.ipynb>), [4.2 Model CNN 1M 4](<4.2 Model CNN 1M 4.ipynb>):** Se introduce dropout en todas las capas.

<br />

- **[4.3 Model TRA 600k](<4.3 Model TRA 600k.ipynb>), [4.3 Model TRA 1M](<4.3 Model TRA 1M.ipynb>):** Modelo basado en Transformers.

<br />

- **[5 Others Evaluations and Errors](<5 Others Evaluations and Errors.ipynb>):** Programa de apoyo para testear los modelos en otros datasets y calcular los errores a partir de evaluaciones pasadas.

## Uso

1. Clona el repositorio (instalar [chocolatey](https://chocolatey.org/install) y luego [git](https://community.chocolatey.org/packages/Git) de no tenerlo):
   ```
   git clone https://github.com/albertnica/TFGF-albertnica
   ```
2. Instala todas las dependencias menos SciServer, la cual debería funcionar abriendo el repositorio clonado en tu IDE (Python 3.13.2):
   ```
   pip install -r requirements.txt
   ```
3. Si se dispone de una gráfica con [soporte CUDA](https://developer.nvidia.com/cuda-gpus) se puede habilitar con:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
4. Descarga los datasets de entrenamiento preprocesados y los espectros de prueba:
   - [/data](https://unioviedo-my.sharepoint.com/:f:/g/personal/uo284776_uniovi_es/EiifkyAiWZ9HqcUwLZppl-4BypXdYoWkSBb3XhzWpCAhqw?e=Zl62pY)
   - [/spectrums](https://unioviedo-my.sharepoint.com/:f:/g/personal/uo284776_uniovi_es/EhBzxXiQb2pOuGz-wNQrxUwBxq-gw6r4Nu20r6s61r3LSg?e=C3JDgY)
   - [/storage](https://unioviedo-my.sharepoint.com/:f:/g/personal/uo284776_uniovi_es/EsZm738_E2FKoC7dOV0W8hIB3JHfctzO4ZNtGTlCgmFDTg?e=Sc1KSx)