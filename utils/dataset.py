"""Data loading utilities for the BUSI dataset.

The BUSI dataset contains, for each class folder, both the raw ultrasound
images and their segmentation masks (files whose name contains "_mask").
Only the raw images are used for classification.
"""
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def get_file_paths(data_dir):
    """Return a list of ``(image_path, class_name)`` tuples, ignoring masks."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}. "
            f"Please download the BUSI dataset and place it under data/."
        )

    samples = []
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            continue
        for name in sorted(os.listdir(class_dir)):
            lower = name.lower()
            if "_mask" in lower:  # skip mask images (e.g. "..._mask.png")
                continue
            if lower.endswith(IMAGE_EXTENSIONS):
                samples.append((class_dir / name, class_name))

    if not samples:
        raise FileNotFoundError(
            f"No images found under {data_dir}. Please check the dataset path."
        )
    return samples


class BUSIDataset(Dataset):
    """Torch Dataset over a list of ``(path, class_name)`` samples."""

    def __init__(self, samples, classes, transform=None):
        self.samples = list(samples)
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.class_to_idx[label]


def build_transforms(img_size):
    """Build train/eval transforms (eval uses only resize + normalize)."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_transform, eval_transform


def stratified_split(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                     seed=42):
    """Split sample indices per class to keep the class balance."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        groups[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for idxs in groups.values():
        idxs = np.asarray(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        n_train = n - n_val - n_test

        test_idx.extend(idxs[:n_test].tolist())
        val_idx.extend(idxs[n_test:n_test + n_val].tolist())
        train_idx.extend(idxs[n_test + n_val:].tolist())

    return train_idx, val_idx, test_idx


def build_loaders(data_dir, img_size=224, batch_size=32, num_workers=4,
                  ratios=(0.7, 0.15, 0.15), seed=42):
    """Build train/val/test DataLoaders and return the ordered class list."""
    samples = get_file_paths(data_dir)
    classes = sorted({label for _, label in samples})

    train_transform, eval_transform = build_transforms(img_size)
    train_idx, val_idx, test_idx = stratified_split(samples, *ratios, seed=seed)

    train_ds = BUSIDataset([samples[i] for i in train_idx], classes,
                           train_transform)
    val_ds = BUSIDataset([samples[i] for i in val_idx], classes, eval_transform)
    test_ds = BUSIDataset([samples[i] for i in test_idx], classes,
                          eval_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, classes
