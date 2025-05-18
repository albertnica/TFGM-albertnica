# Exploración de la utilidad de los modelos MLP para la resolución de EDPs

Este repositorio contiene el código de varios algoritmos de Diferencias Finitas y de Neural Networks que se comparan entre sí con el objetivo de indagar sobre los pros y contras de los métodos convencionales y de los actuales.

## Descripción

Los métodos de resolución de EDPs incluidos en este proyecto son:

- **MDF:** Algoritmos basados en el Método de Diferencias Finitas.
- **NN:** Algoritmos basados en Redes Neuronales, en particular en Multi-Layer Perceptrons.

## Estructura del Proyecto

- **[/models](/models):** Carpeta con los modelos ya entrenados que se presentan en la memoria del trabajo.

<br />

- **[1D_heat_NN](<1D_heat_NN.ipynb>), [1D_heat_MDF](<1D_heat_MDF.ipynb>):** Notebooks dedicados a la resolución de la ecuación del calor en espacio (1D) y tiempo.
- **[1D_heat_alpha_NN](<1D_heat_alpha_NN.ipynb>):** Incorpora el parámetro de difusión térmica al entrenamiento, de forma que lo podemos tratar como una variable más.
- **[1D_heat_NN](<1D_heat_NN.ipynb>), [1D_heat_MDF](<1D_heat_MDF.ipynb>):** Notebooks dedicados a la resolución de la ecuación del calor en espacio (1D) y tiempo.
- **[1D_heat_NN](<1D_heat_NN.ipynb>), [1D_heat_MDF](<1D_heat_MDF.ipynb>):** Notebooks dedicados a la resolución de la ecuación del calor en espacio (1D) y tiempo.

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