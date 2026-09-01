# Laboratorio 1 — Inferencia de una red neuronal con NumPy

**Curso:** IAR401 — Redes Neuronales y Aprendizaje Profundo (UNITEC, Q3 2026)
**Estudiante:** Luis Henríquez

Implementación desde cero, usando únicamente NumPy, del proceso de inferencia
(forward propagation) de un MLP preentrenado para clasificar dígitos manuscritos
del conjunto MNIST. No se entrena ningún modelo ni se usa ningún framework de
deep learning (Keras/TensorFlow/PyTorch/scikit-learn).

## Video de demostración

📹 **Link:** https://youtu.be/82Eemxh1LbE

## Estructura del repositorio

```
Lab1RN_LuisHenriquez/
├── Laboratorio-1-SimpleNN.pdf          # Enunciado del laboratorio
├── Starter-Files/
│   ├── mnist_mlp.json                  # Modelo preentrenado (arquitectura, pesos, sesgos)
│   └── mnist_test.npz                  # Conjunto de prueba MNIST (10,000 imágenes)
└── Lab1_LuisHenriquez/
    ├── Lab1_LuisHenriquez.py           # Implementación
    └── prediccion_individual.png       # Salida: ejemplo de predicción individual
```

## Arquitectura del modelo

| Capa | Operación | Pesos      | Sesgos | Activación |
|------|-----------|------------|--------|------------|
| 1    | Densa     | (784, 128) | (128,) | ReLU       |
| 2    | Densa     | (128, 10)  | (10,)  | Softmax    |

La red se construye dinámicamente a partir de `Starter-Files/mnist_mlp.json` —
las capas, pesos y activaciones no están hardcodeados.

## Cómo ejecutarlo

```bash
cd Lab1_LuisHenriquez
pip install numpy matplotlib
python Lab1_LuisHenriquez.py
```

El script usa rutas relativas a su propia ubicación, así que debe ejecutarse
desde dentro de `Lab1_LuisHenriquez/`.

## Qué hace el script

1. **Carga e inspección del modelo** — lee el JSON, arma una `LayerDense` por
   cada capa y muestra un resumen de la arquitectura.
2. **Construcción de la red** — instancia `NeuralNetwork` con las capas leídas.
3. **Carga y preparación de datos** — lee `mnist_test.npz`, valida formas
   `(10000, 28, 28)` / `(10000,)`, normaliza por 255 y aplana a `(10000, 784)`.
4. **Inferencia y evaluación**:
   - Predicción sobre una sola imagen (clase predicha, etiqueta real,
     probabilidades por clase, imagen guardada en `prediccion_individual.png`).
   - Inferencia vectorizada sobre las 10,000 imágenes y cálculo de accuracy.

Resultado de referencia esperado: **~96.66 % de exactitud**.

## Diseño

- `LayerDense`: guarda `W`, `b` y el nombre de su activación; `forward(A)`
  calcula `Z = A·W + b` y aplica la activación.
- `NeuralNetwork`: contiene la lista de capas en orden y encadena sus `forward`.
- `relu` y `softmax` (estable numéricamente, restando el máximo por fila) están
  registradas en el diccionario `ACTIVACIONES` y se buscan por nombre.

## Restricciones cumplidas

- Sin frameworks de deep learning.
- Sin reentrenamiento ni alteración de pesos/sesgos provistos.
- Arquitectura, pesos y sesgos cargados dinámicamente desde el JSON.
