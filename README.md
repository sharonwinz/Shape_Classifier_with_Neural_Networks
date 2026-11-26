# My Drawing Dataset Classifier

I built a drawing app and a neural network that recognizes four shapes—**circle, triangle, square, and flower**.

This project follows the full process of collecting my own images, training a model, saving its learned weights and biases, and testing it on new drawings.

## How It Works

1. **Collect training drawings**  
   Draw a shape in the app, select its class, and save it as a 28 × 28 grayscale PNG.

2. **Prepare the data**  
   Convert each image into 784 input values and scale the pixel values to between 0 and 1. The image’s folder provides its correct class label.

3. **Train the neural network**  
   Run `train_network.py` to calculate the loss and update the weights and biases over 1,000 training rounds.

4. **Save the model**  
   Store the learned weights, biases, and class names in `trained_model.npz`.

5. **Collect fresh test drawings**  
   Draw new shapes and save them separately under `test_drawings/dataset`.

6. **Test the model**  
   Run `test_network.py` to load the saved model, predict each test drawing’s class, and calculate test accuracy.

## Neural Network

```text
784 input values → 32 hidden neurons → 4 output probabilities
```

The model uses a sigmoid activation in the hidden layer and softmax at the output. Training uses cross-entropy loss and backpropagation.

## Results

### Training

The training dataset contains **215 drawings**.

| Class | Drawings |
|---|---:|
| Circle | 50 |
| Triangle | 65 |
| Square | 50 |
| Flower | 50 |

After 1,000 training rounds, the final loss was **0.0007** and the training accuracy was **100.0%**.

### Testing

On a separate set of **213 fresh drawings**, the model classified **208 correctly**, achieving **97.7% test accuracy**.

Training accuracy measures performance on the images used for learning. Test accuracy measures performance on the separate drawings collected afterward.

## Main Files

| File | Purpose |
|---|---|
| `drawing_collector.py` | Collects and saves labeled training drawings. |
| `prepare_data.py` | Loads training images and prepares their inputs and labels. |
| `neural_network.py` | Demonstrates a forward calculation with initial random weights. |
| `train_network.py` | Trains the model and saves the learned values. |
| `trained_model.npz` | Stores the learned weights, biases, and class names. |
| `test_drawings/drawing_collector.py` | Collects fresh test drawings. |
| `test_network.py` | Loads the saved model and evaluates the test drawings. |

## Setup and Run

This project uses Python, NumPy, Pillow, and Tkinter.

In a Windows terminal opened in the project folder, create an environment and install the dependencies.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy Pillow
```

Collect training drawings.

```powershell
.\.venv\Scripts\python.exe drawing_collector.py
```

Train and save the model.

```powershell
.\.venv\Scripts\python.exe train_network.py
```

Collect new test drawings.

```powershell
.\.venv\Scripts\python.exe .\test_drawings\drawing_collector.py
```

Evaluate the saved model.

```powershell
.\.venv\Scripts\python.exe test_network.py
```
