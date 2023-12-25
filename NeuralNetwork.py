from typing import Any
from Value import Value

import random

class Neuron:

    def __init__(self, num_dimensions) -> None:
        self.weights = [Value(random.uniform(-1,1)) for _ in range(num_dimensions)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        activation = sum((wi*xi for wi, xi in zip(self.weights, x)), self.bias)
        output = activation.tanh()
        return output
    
    def parameters(self):
        return self.weights + [self.bias]
    
class Layer:

    def __init__(self, num_dimensions, num_neurons) -> None:
        self.neurons = [Neuron(num_dimensions) for _ in range(num_neurons)]

    def __call__(self, x):
        outputs = [n(x) for n in self.neurons]
        return outputs[0]  if len(outputs) == 1 else outputs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:

    def __init__(self, num_dimensions, num_neurons_per_layer) -> None:
        size = [num_dimensions] + num_neurons_per_layer
        self.layers = [Layer(size[i], size[i+1]) for i in range(len(num_neurons_per_layer))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]