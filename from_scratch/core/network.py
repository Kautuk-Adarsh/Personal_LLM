# Neuron, Layer, Network — the feedforward part of the model.

import numpy as np

class Neuron:
    def __init__(self, weights: np.ndarray, bias: float):
        self.weights = np.array(weights)
        self.bias = bias

    def relu(self, x: float) -> float:
        return max(0, x)

    def forward(self, inputs) -> float:
        result = np.dot(self.weights, inputs) + self.bias
        return self.relu(result)


class Layer:
    def __init__(self, neurons: list):
        self.neurons = neurons

    def forward(self, inputs) -> list:
        results = []
        for neuron in self.neurons:
            results.append(neuron.forward(inputs))
        return results


class Network:
    def __init__(self, layers: list):
        self.layers = layers

    def forward(self, inputs) -> list:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs