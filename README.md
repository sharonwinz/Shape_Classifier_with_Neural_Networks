# My Drawing Dataset

A small local Python app for collecting **circle, triangle, square, and flower** drawings. Each click of **Save sample** writes a PNG straight to your computer. No chat session or export ZIP is needed.

## 1. Open the project in VS Code

Choose **File → Open Folder** and open:

```text
C:\Users\Sharon\Documents\ChatGPT\AI Class 2
```

Open **Terminal → New Terminal**. Run:

```powershell
py --version
```

If Windows says `py` is not recognized, install Python 3 from [python.org](https://www.python.org/downloads/windows/). In the installer, enable **Add python.exe to PATH**, and keep **Tcl/Tk and IDLE** enabled. Close and reopen VS Code afterward. If `python --version` works but `py` does not, use `python` in place of `py` in the next command.

## 2. One-time setup

Run these commands one line at a time in the project terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

This creates a Python environment just for this project. You do not need to activate it or change PowerShell's execution policy.

If you use VS Code's Python extension, choose **Python: Select Interpreter** from the command palette (`Ctrl+Shift+P`), then select `.venv\Scripts\python.exe`.

## 3. Run the drawing app

```powershell
.\.venv\Scripts\python.exe drawing_collector.py
```

Use that same command every time you return to the project. Keep the terminal open while using the app.

## 4. Test with ONE image first

1. Choose **Flower** and draw a flower by holding the left mouse button.
2. Click **Save sample**. Check that the status says **Saved:** and the Flower counter increases by one.
3. Click **Open dataset folder**, then open `flower`. Confirm there is a PNG file. Open it: the actual image is tiny because it is only 28 × 28 pixels. The app shows an enlarged preview.
4. Close the app and run it again. The Flower counter should still include that sample, and the PNG should still be in the folder.
5. Only after that succeeds, collect a small batch such as 10 drawings per class.

For each new sample: choose its class, draw, save, then clear the canvas. The Save button disables after a successful save to prevent repeated clicks from saving duplicates. Adding strokes lets you save a revised drawing as another sample. Choose the correct class **before** saving; changing the selector does not move an earlier saved file.

**Saving is explicit:** click Save sample for each drawing. Clearing or closing with an unsaved drawing asks whether to save it. If saving fails, the drawing stays on the canvas and you can retry. A forced shutdown can lose a drawing you have not saved yet.

## Where your drawings live

```text
Shape_Neural_Networks/
  drawing_collector.py
  requirements.txt
  dataset/
    circle/
    triangle/
    square/
    flower/
```

The app always uses `dataset` beside `drawing_collector.py`, even if you start it from another folder. Each PNG has a unique filename. Counters are rebuilt from the PNG files when the app starts. Clearing the canvas does not delete saved images. Move or copy the whole project folder if you want to keep the app and dataset together.

Each save first writes and flushes a temporary file, then renames the completed file to `.png`. The app only reports success after that finishes. Keep a second copy of the `dataset` folder on another drive or your usual backup service, especially after a drawing session. Local saving does not protect against drive failure or manual deletion.

## What the images mean for your learning

- Every sample is a **28 × 28 grayscale PNG**: one brightness number per pixel, from 0 (black) to 255 (white).
- The app crops unused black space, fits the drawing within 20 × 20 pixels, and centers it in a 28 × 28 image. This keeps the shape's proportions.
- The folder name is the **label**: circle, triangle, square, or flower.
- Later, NumPy can read these images as 28 × 28 arrays and divide by 255 to get values between 0 and 1. Flattening an image gives 784 input values.
- These are your own four shape classes, not MNIST's ten digit classes. The image dimensions and colors follow an MNIST-style convention. No model training or loss calculation is included yet.

Only the 28 × 28 version is saved; the large drawing canvas is used to make that sample.
