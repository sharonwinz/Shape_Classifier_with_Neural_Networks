#Checks the drawing app’s saving behavior, such as whether a PNG has the right dimensions. 
# This is a separate check from measuring the model’s accuracy.

"""Check image format and persistence without adding user dataset samples."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

from drawing_collector import CLASSES, count_samples, prepare_sample, save_sample


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.drawing = Image.new("L", (336, 336), 0)
        ImageDraw.Draw(self.drawing).ellipse((70, 50, 260, 280), outline=255, width=16)

    def test_save_is_readable_and_counts_survive_reload(self):
        path = save_sample(self.drawing, "flower", self.folder)
        self.assertEqual(path.parent, self.folder / "flower")
        with Image.open(path) as image:
            image.load()
            self.assertEqual(image.format, "PNG")
            self.assertEqual(image.size, (28, 28))
            self.assertEqual(image.mode, "L")
            self.assertEqual(image.getpixel((0, 0)), 0)
            self.assertGreater(image.getextrema()[1], 0)
        # Recount from only the folder, with no in-memory sample list.
        self.assertEqual(count_samples(self.folder), {label: int(label == "flower") for label in CLASSES})

    def test_repeated_saves_do_not_overwrite(self):
        first = save_sample(self.drawing, "circle", self.folder)
        before = first.read_bytes()
        second = save_sample(self.drawing, "circle", self.folder)
        self.assertNotEqual(first, second)
        self.assertEqual(first.read_bytes(), before)
        self.assertEqual(count_samples(self.folder)["circle"], 2)

    def test_blank_and_invalid_labels_do_not_save(self):
        with self.assertRaises(ValueError):
            save_sample(Image.new("L", (336, 336), 0), "square", self.folder)
        with self.assertRaises(ValueError):
            save_sample(self.drawing, "../other", self.folder)
        self.assertEqual(list(self.folder.rglob("*")), [])

    def test_failed_write_leaves_no_finished_sample(self):
        with patch("drawing_collector.os.fsync", side_effect=OSError("Disk write failed")):
            with self.assertRaises(OSError):
                save_sample(self.drawing, "triangle", self.folder)
        self.assertEqual(list((self.folder / "triangle").iterdir()), [])
        self.assertEqual(count_samples(self.folder)["triangle"], 0)

    def test_resize_preserves_proportions_and_padding(self):
        drawing = Image.new("L", (336, 336), 0)
        ImageDraw.Draw(drawing).rectangle((100, 100, 199, 149), fill=255)
        sample = prepare_sample(drawing)
        self.assertEqual(sample.getbbox(), (4, 9, 24, 19))


if __name__ == "__main__":
    unittest.main()
