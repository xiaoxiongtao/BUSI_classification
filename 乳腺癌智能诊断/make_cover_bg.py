"""Create a dark-blue toned ultrasound montage for the courseware cover."""
from PIL import Image, ImageEnhance
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Dataset_BUSI_with_GT"
OUT = ROOT / "乳腺癌智能诊断" / "assets"

W, H = 1280, 720
cols, rows = 5, 3
cw, ch = W // cols, H // rows

canvas = Image.new("RGB", (W, H))
rng = np.random.default_rng(7)
classes = ["benign", "malignant", "normal"]
for r in range(rows):
    for c in range(cols):
        cls = classes[(r + c) % 3]
        imgs = [p for p in sorted((DATA / cls).glob("*.png"))
                if "_mask" not in p.name.lower()]
        im = Image.open(imgs[int(rng.integers(len(imgs)))]).convert("RGB")
        im = im.resize((cw, ch))
        canvas.paste(im, (c * cw, r * ch))

# tone down: grayscale-ish + dark blue overlay
canvas = ImageEnhance.Color(canvas).enhance(0.35)
canvas = ImageEnhance.Brightness(canvas).enhance(0.55)
arr = np.asarray(canvas).astype(np.float32)
overlay = np.array([14, 42, 88], dtype=np.float32)  # #0E2A58-ish navy
arr = arr * 0.35 + overlay * 0.65
Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(OUT / "cover_ultrasound.png")
print("saved", OUT / "cover_ultrasound.png")
