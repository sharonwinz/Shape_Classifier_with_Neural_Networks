#Reads the training PNGs. 
#It turns each drawing into 784 numbers in X and gets its correct label in y from the folder name.

from pathlib import Path

import numpy as np
from PIL import Image


DATASET_DIR = Path(__file__).resolve().parent / "dataset"
CLASSES = ("circle", "triangle", "square", "flower")

inputs = []
labels = []

for label, class_name in enumerate(CLASSES):
    image_files = sorted((DATASET_DIR / class_name).glob("*.png"))

    if not image_files:
        raise SystemExit(f"No PNG drawings found in: {DATASET_DIR / class_name}")

    for image_path in image_files:
        with Image.open(image_path) as image:
            image = image.convert("L")

            if image.size != (28, 28):
                raise SystemExit(
                    f"Expected 28 × 28 pixels: {image_path}"
                )

            pixels = np.array(image, dtype=np.float32)

        # Normalize and flatten each drawing.
        x = (pixels / 255.0).flatten()

        inputs.append(x)
        labels.append(label)

    print(f"{class_name}: {len(image_files)} drawings")

# Combine the drawings and labels into NumPy arrays.
X = np.array(inputs, dtype=np.float32)
y = np.array(labels, dtype=np.int64)

print()
print("Input array shape:", X.shape)
print("Label array shape:", y.shape)
print("First drawing's label:", y[0])