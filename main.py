from Value import Value
from NeuralNetwork import Neuron, Layer, MLP

import math
import numpy as np
import matplotlib.pyplot as plt

#############################
# Simple Backpropogation
#############################

x1 = Value(2.0, label="x1")
x2 = Value(0.0, label="x2")
w1 = Value(-3.0, label="w1")
w2 = Value(1.0, label="w2")
b = Value(6.8813735870195432, label="b")
x1w1 = x1*w1; x1w1.label = "x1w1"
x2w2 = x2*w2; x2w2.label = "x2w2"
x1w1x2w2 = x1w1+x2w2; x1w1x2w2.label = "x1w1 + 2w2"
n = x1w1x2w2 + b; n.label = "n"
e = (2*n).exp()
o = (e-1)/(e+1)
o.label = "output"
o.backward()

#############################
# Neural Network
#############################

n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

y_target = [1.0, -1.0, -1.0, 1.0]

for k in range(20):

    # Forward Pass
    y_pred = [n(x) for x in xs]
    loss = sum([(y_output - y_ground_truth)**2 for y_ground_truth, y_output in zip(y_target, y_pred)])

    # Zero Grad
    for p in n.parameters():
        p.grad = 0.0

    # Backward Pass
    loss.backward()

    # Update
    for p in n.parameters():
        p.data += -0.05 * p.grad

    print("Turn " + str(k) + ", loss: " + str(loss.data))