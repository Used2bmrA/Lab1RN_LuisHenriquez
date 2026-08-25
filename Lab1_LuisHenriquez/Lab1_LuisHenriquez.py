# Laboratorio 1 - Inferencia de una red neuronal con NumPy
# Luis Henríquez - 10941366

import json                             # Lectura del JSON con pesos entrenados
import os                               # Path de archivos
import matplotlib.pyplot as plt         # Visualización de imágenes
import numpy as np                      # Cálculos matriciales

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                             # Path del script actual
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "Starter-Files", "mnist_mlp.json")      # Path del JSON con pesos entrenados
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Starter-Files", "mnist_test.npz")       # Path del dataset de prueba (imágenes y etiquetas)

# ---------------------------------------------------------------------------
# Capa densa
# ---------------------------------------------------------------------------
class LayerDense:
    
    def __init__(self, W, b):
        self.weights = np.asarray(W, dtype=np.float64)
        self.biases = np.asarray(b, dtype=np.float64)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

    
# ---------------------------------------------------------------------------
# Funciones de activacion
# ---------------------------------------------------------------------------
class ActivationReLU:
    def forward(self, input):
        self.output = np.maximum(0, input)
        return self.output


class ActivationSoftmax:
    def forward(self, input):
        max = np.max(input, axis=1, keepdims=True)
        shifted = input - max
        exp_values = np.exp(shifted)
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.output

def main():
    # 1) Cargar e inspeccionar el modelo
    print("=" * 70)
    print("1) Cargando modelo desde JSON")
    print("=" * 70)
    

if __name__ == "__main__":
    main()
