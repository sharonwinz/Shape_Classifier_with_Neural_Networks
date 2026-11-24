"""Collect local, labeled drawings. Run with: python drawing_collector.py"""

from pathlib import Path
import os
import tempfile
import uuid
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

CLASSES = ("circle", "triangle", "square", "flower")
DATASET_DIR = Path(__file__).resolve().parent / "dataset"
CANVAS_SIZE = 336
BRUSH_WIDTH = 16


def prepare_sample(drawing):
    """Return 28x28 grayscale: bright strokes on black, like MNIST.

    Crop excess blank space, fit inside 20x20, then center with padding.
    The original drawing already uses white strokes on a black background.
    """
    drawing = drawing.convert("L")
    bounds = drawing.getbbox()
    if bounds is None:
        raise ValueError("Draw something before saving.")
    cropped = drawing.crop(bounds)
    cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
    sample = Image.new("L", (28, 28), 0)
    sample.paste(cropped, ((28 - cropped.width) // 2, (28 - cropped.height) // 2))
    return sample


def save_sample(drawing, label, dataset_dir=DATASET_DIR):
    """Flush a complete PNG to disk before publishing its unique filename."""
    if label not in CLASSES:
        raise ValueError("Choose one of the four drawing classes.")
    sample = prepare_sample(drawing)
    folder = Path(dataset_dir) / label
    folder.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex
    destination = folder / (name + ".png")
    # A temporary file in the same folder allows an atomic rename. A failed
    # save never leaves a partial PNG counted as a finished sample.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=folder, suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            sample.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def count_samples(dataset_dir=DATASET_DIR):
    return {label: sum(1 for path in (Path(dataset_dir) / label).glob("*.png") if path.is_file())
            for label in CLASSES}


class DrawingCollector:
    def __init__(self, root):
        self.root = root
        self.dirty = False
        self.has_drawing = False
        self.previous_point = None
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.label = tk.StringVar(value=CLASSES[0])
        self.status = tk.StringVar(value="First: draw ONE sample, save it, then open the dataset folder to check it.")
        self.counts = tk.StringVar()
        root.title("My Drawing Dataset")
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", self.close)

        frame = ttk.Frame(root, padding=18)
        frame.grid()
        ttk.Label(frame, text="My Drawing Dataset", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(frame, text="1. Choose a class   2. Draw   3. Save sample").grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 12))
        choices = ttk.Frame(frame)
        choices.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 12))
        for label in CLASSES:
            ttk.Radiobutton(choices, text=label.title(), value=label, variable=self.label).pack(side="left", padx=(0, 12))

        self.canvas = tk.Canvas(frame, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="black", highlightthickness=1, highlightbackground="#888888")
        self.canvas.grid(row=3, column=0, sticky="w")
        self.canvas.bind("<Button-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw_stroke)
        root.bind("<ButtonRelease-1>", self.end_stroke)

        side = ttk.Frame(frame, padding=(18, 0, 0, 0))
        side.grid(row=3, column=1, sticky="n")
        ttk.Label(side, text="Saved image (28 × 28)").pack()
        self.preview = ttk.Label(side)
        self.preview.pack(pady=8)
        ttk.Label(side, text="Preview enlarged 5×\nWhite ink, black background", justify="center").pack()
        ttk.Label(side, textvariable=self.counts, justify="left").pack(pady=18, anchor="w")

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=2, sticky="w", pady=12)
        self.save_button = ttk.Button(buttons, text="Save sample", command=self.save, state="disabled")
        self.save_button.pack(side="left")
        ttk.Button(buttons, text="Clear canvas", command=self.clear).pack(side="left", padx=10)
        ttk.Button(buttons, text="Open dataset folder", command=self.open_folder).pack(side="left")
        ttk.Label(frame, textvariable=self.status, wraplength=550, justify="left").grid(row=5, column=0, columnspan=2, sticky="w")
        ttk.Label(frame, text="Saves stay on your computer after closing this app.\nClear canvas does not delete saved files.", foreground="#555555").grid(row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.refresh_counts()
        self.update_preview()

    def point(self, event):
        return (max(0, min(CANVAS_SIZE - 1, event.x)),
                max(0, min(CANVAS_SIZE - 1, event.y)))

    def start_stroke(self, event):
        x, y = self.point(event)
        radius = BRUSH_WIDTH // 2
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="white", outline="white")
        self.pen.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
        self.previous_point = (x, y)
        self.mark_dirty()

    def draw_stroke(self, event):
        if self.previous_point is None:
            return
        point = self.point(event)
        x, y = point
        radius = BRUSH_WIDTH // 2
        self.canvas.create_line(*self.previous_point, *point, fill="white", width=BRUSH_WIDTH, capstyle=tk.ROUND)
        self.canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill="white", outline="white")
        self.pen.line((self.previous_point, point), fill=255, width=BRUSH_WIDTH)
        self.pen.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
        self.previous_point = point
        self.mark_dirty()

    def mark_dirty(self):
        self.dirty = True
        self.has_drawing = True
        self.save_button.configure(state="normal")
        self.status.set("Unsaved drawing — click Save sample to keep it on disk.")

    def end_stroke(self, _event):
        self.previous_point = None
        self.update_preview()

    def update_preview(self):
        sample = prepare_sample(self.image) if self.has_drawing else Image.new("L", (28, 28), 0)
        self.preview_image = ImageTk.PhotoImage(sample.resize((140, 140), Image.Resampling.NEAREST))
        self.preview.configure(image=self.preview_image)

    def refresh_counts(self):
        counts = count_samples()
        self.counts.set("Saved on disk\n\n" + "\n".join(f"{label.title()}: {counts[label]}" for label in CLASSES)
                        + f"\n\nTotal: {sum(counts.values())}")

    def save(self):
        if not self.dirty:
            return False
        try:
            path = save_sample(self.image, self.label.get())
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not save", f"Your drawing is still on the canvas.\n\n{error}", parent=self.root)
            return False
        self.dirty = False
        self.save_button.configure(state="disabled")
        self.refresh_counts()
        self.status.set(f"Saved: {path}\nClear the canvas when you are ready for the next drawing.")
        return True

    def okay_to_discard(self):
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel("Unsaved drawing", "Save this drawing before continuing?\n\nYes = save   No = discard   Cancel = keep drawing", parent=self.root)
        if answer is None:
            return False
        return self.save() if answer else True

    def clear(self):
        if not self.okay_to_discard():
            return
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.pen = ImageDraw.Draw(self.image)
        self.has_drawing = False
        self.dirty = False
        self.previous_point = None
        self.save_button.configure(state="disabled")
        self.update_preview()
        self.status.set("Canvas cleared. Your saved PNG files are still in the dataset folder.")

    def open_folder(self):
        try:
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(str(DATASET_DIR))
        except OSError as error:
            messagebox.showerror("Could not open folder", str(error), parent=self.root)

    def close(self):
        if self.okay_to_discard():
            self.root.destroy()


def main():
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise SystemExit(
            "The drawing window could not open. Install or repair Python with "
            "Tcl/Tk enabled (see README.md), then try again.\n\n" + str(error)
        ) from error
    try:
        for label in CLASSES:
            (DATASET_DIR / label).mkdir(parents=True, exist_ok=True)
        DrawingCollector(root)
    except OSError as error:
        messagebox.showerror("Cannot access dataset folder", f"{DATASET_DIR}\n\n{error}", parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
