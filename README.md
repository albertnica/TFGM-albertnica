# Exploración de la utilidad de las RNAs para la resolución de EDPs

Este repositorio contiene el código de varios algoritmos basados en el Método de Diferencias Finitas y en Neural Networks que se comparan entre sí con el objetivo de indagar sobre los pros y contras de los métodos convencionales y de los actuales. Un análisis detallado se presenta en la [memoria](<TFG_Matematicas_Alberto_Nieto_Cardoso.pdf>) correspondiente.

## Descripción

Los métodos de resolución de EDPs incluidos en este proyecto son:

- **_MDF:** Algoritmos basados en el Método de Diferencias Finitas.
- **_NN:** Algoritmos basados en Redes Neuronales, en particular en Multi-Layer Perceptrons.

## Estructura del Proyecto

- **[/models](/models):** Carpeta con los modelos ya entrenados que se presentan en la memoria del trabajo.

<br />

- **[1 heat_1D_NN](<1 heat_1D_NN.ipynb>), [1 1D_heat_MDF](<1 heat_1D_MDF.ipynb>):** Notebooks dedicados a la resolución de la ecuación del calor en espacio (1D) y tiempo.
- **[2 poisson_2D_NN](<2 poisson_2D_NN.ipynb>), [2 poisson_2D_MDF](<2 poisson_2D_MDF.ipynb>):** Notebooks dedicados a la resolución de la ecuación del calor en espacio (1D) y tiempo.
- **[3 taylor_green_2D_NN](<3 taylor_green_2D_NN.ipynb>), [3 taylor_green_2D_MDF](<3 taylor_green_2D_MDF.m>):** Notebooks dedicados a la resolución de la ecuación del calor en espacio (1D) y tiempo.
- **[4 heat_alpha_1D_NN](<4 heat_alpha_1D_NN.ipynb>):** Incorpora el parámetro de difusión térmica al entrenamiento, de forma que lo podemos tratar como una variable más.
- **[5 heat_2D_NN](<5 heat_2D_NN.ipynb>):** Variante del caso bidimensional sobre dominio triangular.

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