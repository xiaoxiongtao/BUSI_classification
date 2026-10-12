"""Build a small set of illustration images used in the lecture slides."""
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Dataset_BUSI_with_GT"
OUT = ROOT / "outputs" / "ppt_assets"
OUT.mkdir(parents=True, exist_ok=True)

CLASSES = ["benign", "malignant", "normal"]
CN = {"benign": "良性 benign", "malignant": "恶性 malignant", "normal": "正常 normal"}


def list_images(cls):
    paths = sorted((DATA / cls).glob("*.png"))
    return [p for p in paths if "_mask" not in p.name.lower()]


# ---- 1. sample grid: 3 rows (class) x 4 cols ----
cell = 260
pad = 12
label_h = 40
rows, cols = 3, 4
grid = Image.new("RGB", (cols * cell + (cols + 1) * pad,
                         rows * (cell + label_h) + (rows + 1) * pad),
                 (247, 249, 252))
draw = ImageDraw.Draw(grid)
rng = np.random.default_rng(0)
for r, cls in enumerate(CLASSES):
    imgs = list_images(cls)
    picks = rng.choice(len(imgs), cols, replace=False)
    for c, idx in enumerate(picks):
        im = Image.open(imgs[idx]).convert("RGB")
        im = im.resize((cell, cell))
        x = pad + c * (cell + pad)
        y = pad + r * (cell + label_h + pad)
        grid.paste(im, (x, y))
        draw.text((x + 6, y + cell + 10), CN[cls], fill=(30, 41, 59))
grid.save(OUT / "dataset_samples.png")
print("counts:", {c: len(list_images(c)) for c in CLASSES})

# ---- 2. single example images (one per class, large) ----
for cls in CLASSES:
    im = Image.open(list_images(cls)[0]).convert("RGB")
    im.thumbnail((420, 420))
    im.save(OUT / f"sample_{cls}.png")

print("saved to", OUT)
