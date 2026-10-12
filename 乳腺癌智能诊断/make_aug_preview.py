"""Render a 2x3 preview grid of the BUSI training augmentations."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Dataset_BUSI_with_GT"
OUT = ROOT / "乳腺癌智能诊断" / "assets"

src = sorted((DATA / "malignant").glob("*.png"))
src = [p for p in src if "_mask" not in p.name.lower()][0]
img = Image.open(src).convert("RGB")

CELL = 250
LAB = 34
PAD = 8
rng_state = 1234

tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
])

import random
import torch
random.seed(rng_state); torch.manual_seed(rng_state)

variants = [("原图", img)]
names = ["随机裁剪+翻转", "水平翻转", "垂直翻转", "旋转 20°", "色彩抖动"]
single = {
    "随机裁剪+翻转": transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    "水平翻转": transforms.RandomHorizontalFlip(p=1.0),
    "垂直翻转": transforms.RandomVerticalFlip(p=1.0),
    "旋转 20°": transforms.RandomRotation(degrees=20),
    "色彩抖动": transforms.ColorJitter(brightness=0.3, contrast=0.3),
}
for n in names:
    random.seed(rng_state); torch.manual_seed(rng_state)
    variants.append((n, single[n](img)))

W = 3 * CELL + 4 * PAD
H = 2 * (CELL + LAB) + 3 * PAD
canvas = Image.new("RGB", (W, H), (247, 249, 252))
draw = ImageDraw.Draw(canvas)
for i, (name, im) in enumerate(variants):
    r, c = divmod(i, 3)
    im2 = im.resize((CELL, CELL))
    x = PAD + c * (CELL + PAD)
    y = PAD + r * (CELL + LAB + PAD)
    canvas.paste(im2, (x, y))
    draw.text((x + 8, y + CELL + 8), name, fill=(30, 41, 59))

canvas.save(OUT / "augment_preview.png")
print("saved", OUT / "augment_preview.png", canvas.size)
