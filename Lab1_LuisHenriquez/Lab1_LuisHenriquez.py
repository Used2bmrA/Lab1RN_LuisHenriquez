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

# ---------------------------------------------------------------------------
# Construccion de la red neuronal
# ---------------------------------------------------------------------------
ACTIVACIONES = {                     #Tipo de activaciones disponibles
    "relu": ActivationReLU,
    "softmax": ActivationSoftmax,
}

def cargarModelo(path):
    # Lee el JSON con pesos entrenados y arma la lista de capas.
    # Devuelve un dict con tres claves: "layers", "input_shape" y "scale".
    # "layers" es una lista de objetos LayerDense y Activation en orden.
    # "input_shape" es una tupla (n_entrada,) y "scale" es un float.
    
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    
    input_shape = tuple(config["input_shape"])
    scale = float(config["preprocess"]["scale"])

    layers = []                                                     # Lista de capas (pesos, sesgos y activaciones)
    for layer_cfg in config["layers"]:
        layers.append(LayerDense(layer_cfg["W"], layer_cfg["b"]))   # Pesos y sesgo
        layers.append(ACTIVACIONES[layer_cfg["activation"]]())      # Función de activación

    return layers, input_shape, scale

def resumenModelo(layers, input_shape, scale):
    print("Arquitectura de la red:")
    print(f"  input_shape = {input_shape}, scale = {scale}")
    dense_layers = [layer for layer in layers if isinstance(layer, LayerDense)]
    activations = [layer for layer in layers if not isinstance(layer, LayerDense)]
    for i, (dense, activation) in enumerate(zip(dense_layers, activations), start=1):
        print(f"  Capa {i}: tipo=dense, unidades={dense.weights.shape[1]}, "
              f"activacion={type(activation).__name__}, "
              f"W={dense.weights.shape}, b={dense.biases.shape}")

def main():
    # 1) Cargar e inspeccionar el modelo
    print("=" * 70)
    print("1) Cargando modelo desde JSON")
    print("=" * 70)
    layers, input_shape, scale = cargarModelo(MODEL_PATH)
    resumenModelo(layers, input_shape, scale)

if __name__ == "__main__":
    main()
