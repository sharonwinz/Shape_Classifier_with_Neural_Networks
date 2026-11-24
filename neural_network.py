#Takes one training image and makes a guess using the initial random weights 

import numpy as np

from prepare_data import X, y
from sigmoid_function import sigmoid
from softmax_function import softmax


# Starting values for the network.
# The seed makes the random values repeat when you rerun the code.
rng = np.random.default_rng(42)

# First layer: 784 inputs → 32 hidden neurons
W1 = rng.normal(0, 0.01, size=(784, 32))
b1 = np.zeros(32)

# Second layer: 32 hidden neurons → 4 outputs
W2 = rng.normal(0, 0.01, size=(32, 4))
b2 = np.zeros(4)


# Use one drawing first.
index = 55

x = X[index]   
correct_label = y[index]

# Forward calculation
a1 = np.dot(x, W1) + b1
z1 = sigmoid(a1)

a2 = np.dot(z1, W2) + b2
probabilities = softmax(a2)


# Display the results.
classes = ("circle", "triangle", "square", "flower")

for class_name, probability in zip(classes, probabilities):
    print(f"{class_name}: {probability:.2%}")

predicted_label = np.argmax(probabilities)

print()
print("Correct class:", classes[correct_label])
print("Predicted class:", classes[predicted_label])
print("Probability sum:", probabilities.sum())
