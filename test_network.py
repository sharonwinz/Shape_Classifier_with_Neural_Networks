#Loads those learned values, guesses the class of each test drawing, 
#and compares the guesses with the folder labels to calculate test accuracy.

from pathlib import Path

import numpy as np
from PIL import Image


project = Path(__file__).resolve().parent
test_folder = project / "test_drawings" / "dataset"

with np.load(project / "trained_model.npz") as model:
    W1, b1 = model["W1"], model["b1"]
    W2, b2 = model["W2"], model["b2"]
    classes = model["classes"]

correct = 0
total = 0

for actual_class in classes:
    for image_path in sorted((test_folder / actual_class).glob("*.png")):
        with Image.open(image_path) as image:
            pixels = np.array(image.convert("L"), dtype=np.float32)
        x = pixels.flatten() / 255.0

        # Same forward calculation, now with the learned weights.
        hidden = 1 / (1 + np.exp(-(x @ W1 + b1)))
        scores = hidden @ W2 + b2
        exp_scores = np.exp(scores - scores.max())
        probabilities = exp_scores / exp_scores.sum()

        predicted_index = np.argmax(probabilities)
        predicted_class = classes[predicted_index]

        print(
            f"Actual: {actual_class:8s} | "
            f"Guess: {predicted_class:8s} | "
            f"Guess probability: {probabilities[predicted_index]:.1%}"
        )

        correct += int(predicted_class == actual_class)
        total += 1

if total == 0:
    raise SystemExit(f"No test drawings found in {test_folder}")

print(f"\nTest accuracy: {correct}/{total} = {correct / total:.1%}")