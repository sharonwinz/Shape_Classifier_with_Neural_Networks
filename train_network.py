#Uses training drawings to calculate loss and adjust the weights and biases. 
#It saves the learned values in trained_model.npz

from pathlib import Path
import numpy as np
from prepare_data import X, y

# Functions that work on a whole batch of drawings.
def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def softmax(a):
    # Each row represents one drawing.
    shifted = a - np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(shifted)
    return exp_a / np.sum(exp_a, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    z1 = sigmoid(X @ W1 + b1)
    scores = z1 @ W2 + b2
    probabilities = softmax(scores)
    return z1, scores, probabilities

def calculate_loss(scores, y):
    # Numerically stable cross-entropy loss.
    log_sum_exp = np.logaddexp.reduce(scores, axis=1)
    correct_scores = scores[np.arange(len(y)), y]
    return np.mean(log_sum_exp - correct_scores)

# 1. Create the same network: 784 → 32 → 4.
rng = np.random.default_rng(42)

W1 = rng.normal(0, 0.01, size=(784, 32))
b1 = np.zeros(32)

W2 = rng.normal(0, 0.01, size=(32, 4))
b2 = np.zeros(4)


# 2. Choose training settings.
learning_rate = 1.0
epochs = 1000

number_of_images = len(X)


# 3. Train using all drawings.
for epoch in range(epochs):

    # A. Forward calculation
    z1, scores, probabilities = forward(X, W1, b1, W2, b2)

    loss = calculate_loss(scores, y)

    if epoch % 100 == 0:
        predictions = np.argmax(probabilities, axis=1)
        accuracy = np.mean(predictions == y)

        print(
            f"Before update {epoch:4d} | "
            f"Loss: {loss:.4f} | "
            f"Training accuracy: {accuracy:.1%}"
        )

    # B. Backpropagation: calculate gradients.
    # Softmax + cross-entropy gradient
    d_scores = probabilities.copy()
    d_scores[np.arange(number_of_images), y] -= 1
    d_scores /= number_of_images

    # Gradients for the second layer
    dW2 = z1.T @ d_scores
    db2 = np.sum(d_scores, axis=0)

    # Pass the gradient back through the sigmoid.
    d_z1 = d_scores @ W2.T
    d_a1 = d_z1 * z1 * (1 - z1)

    # Gradients for the first layer
    dW1 = X.T @ d_a1
    db1 = np.sum(d_a1, axis=0)

    # C. Update all weights and biases.
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


# 4. Check the final trained network.
_, scores, probabilities = forward(X, W1, b1, W2, b2)

final_loss = calculate_loss(scores, y)
predictions = np.argmax(probabilities, axis=1)
training_accuracy = np.mean(predictions == y)

print()
print("Training finished!")
print(f"Final loss: {final_loss:.4f}")
print(f"Final training accuracy: {training_accuracy:.1%}")


# 5. Save the learned weights and biases.
model_path = Path(__file__).resolve().parent / "trained_model.npz"

np.savez(
    model_path,
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2,
    classes=np.array(["circle", "triangle", "square", "flower"]),
)

print("Saved model:", model_path)