"""
Write a program to demonstrate the working of different activation functions like Sigmoid, Tanh, RELU and softmax to train neural network.
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / np.sum(e_x, axis=0)

x = np.linspace(-10, 10, 400)

y = {
    'sigmoid': sigmoid(x),
    'tanh': tanh(x),
    'relu': relu(x),
    'softmax': softmax(np.array([x, x * 0.5, x * 0.2])).T
}

plt.figure(figsize=(12, 8))
titles = ["Sigmoid", "Tanh", "ReLU", "Softmax"]
colors = ['blue', 'orange', 'green']
markers = ['-', '--', '-.']

for i, (key, y_values) in enumerate(y.items()):
    plt.subplot(2, 2, i + 1)
    if key == 'softmax':
        for j in range(y_values.shape[1]):
            plt.plot(x, y_values[:, j], label=f"Set {j+1}", linestyle=markers[j])
    else:
        plt.plot(x, y_values, label=key, color=colors[i])
    plt.title(f"{titles[i]} Activation Function")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()
