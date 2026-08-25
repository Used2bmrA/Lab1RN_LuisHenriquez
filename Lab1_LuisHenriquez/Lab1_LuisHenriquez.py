# Laboratorio 1 - Inferencia de una red neuronal con NumPy
# Luis Henríquez - 10941366

import json                             # Lectura del JSON con pesos entrenados
import os                               # Path de archivos
import matplotlib.pyplot as plt         # Visualización de imágenes
import numpy as np                      # Cálculos matriciales

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))                             # Path del script actual
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "Starter-Files", "mnist_mlp.json")      # Path del JSON con pesos entrenados
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "Starter-Files", "mnist_test.npz")       # Path del dataset de prueba (imágenes y etiquetas)


def main():
    # 1) Cargar e inspeccionar el modelo
    print("=" * 70)
    print("1) Cargando modelo desde JSON")
    print("=" * 70)
    

if __name__ == "__main__":
    main()
