# Exploración de la utilidad de las RNAs para la resolución de EDPs

Este repositorio contiene el código de varios algoritmos basados en el Método de Diferencias Finitas y en Neural Networks que se comparan entre sí con el objetivo de indagar sobre los pros y contras de los métodos convencionales y de los actuales.

## Descripción

Los métodos de resolución de EDPs incluidos en este proyecto son:

- **_MDF:** Algoritmos basados en el Método de Diferencias Finitas.
- **_NN:** Algoritmos basados en Redes Neuronales, en particular en Multi-Layer Perceptrons.

## Estructura del Proyecto

- **[/models](/models):** Carpeta con los modelos ya entrenados que se presentan en la memoria del trabajo.

<br />

- **[heat_1D_NN](<heat_1D_NN.ipynb>), [1D_heat_MDF](<heat_1D_MDF.ipynb>):** Notebooks dedicados a la resolución de la ecuación del calor en espacio (1D) y tiempo.
- **[heat_alpha_1D_NN](<heat_alpha_1D_NN.ipynb>):** Incorpora el parámetro de difusión térmica al entrenamiento, de forma que lo podemos tratar como una variable más.
- **[heat_2D_NN](<heat_2D_NN.ipynb>):** Variante del caso bidimensional sobre dominio triangular.
- **[poisson_2D_NN](<poisson_2D_NN.ipynb>), [poisson_2D_MDF](<poisson_2D_MDF.ipynb>):** Notebooks dedicados a la resolución de la ecuación del calor en espacio (1D) y tiempo.
- **[taylor_green_2D_NN](<taylor_green_2D_NN.ipynb>), [taylor_green_2D_MDF](<taylor_green_2D_MDF.m>):** Notebooks dedicados a la resolución de la ecuación del calor en espacio (1D) y tiempo.

## Uso

1. Clona el repositorio (instalar [chocolatey](https://chocolatey.org/install) y luego [git](https://community.chocolatey.org/packages/Git) de no tenerlo):
   ```
   git clone https://github.com/albertotfgm/Alberto-Nieto-Cardoso-TFGM
   ```
2. Instala todas las dependencias (Python 3.13.3):
   ```
   pip install -r requirements.txt
   ```
3. Si se dispone de una gráfica con [soporte CUDA](https://developer.nvidia.com/cuda-gpus) se puede habilitar con:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```