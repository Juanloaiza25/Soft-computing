"""
SnakeAI — Red Neuronal Feedforward (NumPy)
============================================
Traducción fiel de NeuralNet.pde + Matrix.pde.

Arquitectura: 24 → [16]×2 → 4
  - Bias nodes incluidos (como el original)
  - Activación ReLU
  - Mutación gaussiana con clamp a [-1, 1]
  - Crossover de un punto (por matriz)
"""

import numpy as np
from copy import deepcopy
from utils.constants import (
    NN_INPUT_SIZE, NN_HIDDEN_NODES,
    NN_HIDDEN_LAYERS, NN_OUTPUT_SIZE, MUTATION_RATE,
)


class NeuralNetwork:
    """
    Red neuronal idéntica al Processing original.
    Pesos incluyen bias (+1 columna por capa).
    """

    def __init__(self, i_nodes=NN_INPUT_SIZE, h_nodes=NN_HIDDEN_NODES,
                 o_nodes=NN_OUTPUT_SIZE, h_layers=NN_HIDDEN_LAYERS):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes
        self.h_layers = h_layers

        # weights[0]: (h_nodes, i_nodes+1) — input→hidden con bias
        # weights[1..h_layers-1]: (h_nodes, h_nodes+1) — hidden→hidden con bias
        # weights[-1]: (o_nodes, h_nodes+1) — hidden→output con bias
        self.weights = []

        # Primera capa: input → hidden
        self.weights.append(np.random.randn(h_nodes, i_nodes + 1) * 0.3)

        # Capas ocultas intermedias
        for _ in range(1, h_layers):
            self.weights.append(np.random.randn(h_nodes, h_nodes + 1) * 0.3)

        # Última capa: hidden → output
        self.weights.append(np.random.randn(o_nodes, h_nodes + 1) * 0.3)

    def output(self, input_arr):
        """
        Propagación hacia adelante (igual que NeuralNet.output en Processing).
        input_arr: array de 26 floats.
        Retorna array de 3 floats.
        Almacena activaciones por capa para visualización.
        """
        # Convertir input a columna
        curr = np.array(input_arr, dtype=np.float64).reshape(-1, 1)

        # Guardar activaciones para visualización
        self.activations = [curr.flatten().copy()]

        # Añadir bias
        curr = np.vstack([curr, [[1.0]]])

        for i in range(self.h_layers):
            hidden = self.weights[i] @ curr           # dot product
            hidden = np.tanh(hidden)           # tanh
            self.activations.append(hidden.flatten().copy())
            curr = np.vstack([hidden, [[1.0]]])       # add bias

        # Capa de salida
        out = self.weights[-1] @ curr
        self.activations.append(out.flatten().copy())
        return out.flatten()

    def predict(self, vision):
        """Retorna el índice de la acción con mayor valor."""
        decision = self.output(vision)
        return int(np.argmax(decision))

    def get_flat_weights(self):
        return np.concatenate([w.ravel() for w in self.weights])

    def set_flat_weights(self, flat):
        idx = 0
        for i in range(len(self.weights)):
            size = self.weights[i].size
            self.weights[i] = flat[idx:idx + size].reshape(
                self.weights[i].shape
        ).copy()
        idx += size

    def chromosome_size(self):
        return sum(w.size for w in self.weights)



