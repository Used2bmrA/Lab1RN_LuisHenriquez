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

ACTIVACIONES = {                     #Tipo de activaciones disponibles
    "relu": ActivationReLU,
    "softmax": ActivationSoftmax,
}

# ---------------------------------------------------------------------------
# Construccion de la red neuronal
# ---------------------------------------------------------------------------
def cargarModelo(path):
    # Lee el JSON con pesos entrenados y arma la lista de capas.
    # Devuelve un dict con tres claves: "layers", "input_shape" y "scale".
    # "layers" es una lista de objetos LayerDense y Activation en orden.
    # "input_shape" es una tupla (n_entrada,) y "scale" es un float.
    
    with open(path, encoding="utf-8") as f:
        modelo_json = json.load(f)

    input_shape = tuple(modelo_json["input_shape"])
    scale = float(modelo_json["preprocess"]["scale"])

    layers = []                                                     # Lista de capas (pesos, sesgos y activaciones)
    for layer_cfg in modelo_json["layers"]:
        layers.append(LayerDense(layer_cfg["W"], layer_cfg["b"]))   # Pesos y sesgo
        layers.append(ACTIVACIONES[layer_cfg["activation"]]())      # Función de activación

    return layers, input_shape, scale

def resumenModelo(layers, input_shape, scale):
    print("Arquitectura de la red:")
    print(f"  input_shape = {input_shape}, scale = {scale}")
    dense_layers = [layer for layer in layers if isinstance(layer, LayerDense)]
    activations = [layer for layer in layers if not isinstance(layer, LayerDense)]
    for i, (dense, activation) in enumerate(zip(dense_layers, activations), start=1):   # Comienza desde 1 porque la capa 0 es la de entrada
        print(f"  Capa {i}: tipo=dense, unidades={dense.weights.shape[1]}, "
              f"activacion={type(activation).__name__}, "
              f"W={dense.weights.shape}, b={dense.biases.shape}")

class NeuralNetwork:
    def __init__(self, layers: list):
        self.layers = layers                    # lista de capas y activaciones en orden

    def forward(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

# ---------------------------------------------------------------------------
# Preprocesamiento de datos
# ---------------------------------------------------------------------------
def load_test_data(path, scale):
    data = np.load(path)
    images, labels = data["images"], data["labels"]

    #print(data.files)                                  # ['images', 'labels']
    #for key in data.files:
    #   print(key, data[key].shape, data[key].dtype)    # images (10000, 28, 28) uint8, labels (10000,) uint8

    assert images.shape == (10000, 28, 28), f"forma inesperada de images: {images.shape}"
    assert labels.shape == (10000,), f"forma inesperada de labels: {labels.shape}"

    X = images.astype(np.float64) / scale          # normalizacion [0,1]
    X = X.reshape(X.shape[0], -1)                  # aplanamiento -> (10000, 784)
    y = labels.astype(np.int64)                    # 10,000 etiquetas como enteros (0-9)
    return X, y

def main():
    # 1) Cargar e inspeccionar el modelo
    print("=" * 70)
    print("1) Cargar e inspeccionar el modelo")
    print("=" * 70)
    layers, input_shape, scale = cargarModelo(MODEL_PATH)
    resumenModelo(layers, input_shape, scale)

    # 2) Implementar la red neuronal
    print("\n" + "=" * 70)
    print("2) Implementar la red neuronal")
    print("=" * 70)
    red = NeuralNetwork(layers)             # A partir de las capas leidas del JSON crea un objeto NeuralNetwork
    print("Red neuronal construida a partir de las capas leidas del JSON.")

    # 3) Cargar y preparar el conjunto de prueba
    print("\n" + "=" * 70)
    print("3) Cargar y preparar el conjunto de prueba")
    print("=" * 70)
    X, y = load_test_data(DATA_PATH, scale)
    print("'X' es un arreglo de imágenes aplanadas y 'y' es un arreglo de etiquetas sobre el número que representa la imagen (0-9).")
    print(f"  X: {X.shape}, dtype={X.dtype}")
    print(f"  y: {y.shape}, dtype={y.dtype}")

    # 4) Ejecutar inferencia y evaluar
    print("\n" + "=" * 70)
    print("4) Ejecutar inferencia y evaluar")
    print("=" * 70)

    print("4.1) Prediccion individual: se le da a la red 1 imagen para")
    print("     verificar dimensiones de la salida y probabilidades de la inferencia.")
    muestraIndividual = X[0:1]
    probsMuestra = red.forward(muestraIndividual)
    clasePredicha = np.argmax(probsMuestra[0])   # indice de la mayor probabilidad = digito predicho
    print(f"  Clase predicha: {clasePredicha}")
    print(f"  Etiqueta real: {y[0]}")
    print("  Probabilidades por clase:")
    for clase in range(10):
        print(f"    clase {clase}: {probsMuestra[0, clase]:.4f}")
    print(f"  Suma de probabilidades: {probsMuestra[0].sum():.6f}")


    print("\n" + "=" * 70)
    print("4.2) Inferencia sobre las 10,000 imágenes de prueba y cálculo de exactitud")
    print("=" * 70)
    probsTotal = red.forward(X)
    print(f"  Forma de la salida de todo el conjunto: {probsTotal.shape}")
    sumas = probsTotal.sum(axis=1)
    print(f"  Suma minima: {sumas.min():.6f}, suma maxima: {sumas.max():.6f}")

    y_pred = np.argmax(probsTotal, axis=1)
    accuracy = np.mean(y_pred == y) # proporcion de aciertos para las 10,000 imagenes
    print(f"\n  Exactitud (accuracy) para el conjunto de prueba: {accuracy * 100:.2f} %")

    images_raw = np.load(DATA_PATH)["images"]

if __name__ == "__main__":
    main()
