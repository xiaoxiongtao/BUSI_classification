"""Visualization helpers for dataset exploration and training results."""
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(tensor):
    """Invert ImageNet normalization so an image can be displayed."""
    img = tensor.detach().cpu().clone()
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype).view(3, 1, 1)
    img = img * std + mean
    return img.clamp(0, 1)


def plot_class_distribution(counts, classes, title="Class Distribution",
                            save_path=None):
    """Bar plot of the number of images per class."""
    values = [counts.get(c, 0) for c in classes]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(classes, values, color=sns.color_palette("muted", len(classes)))
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(v),
                ha="center", va="bottom", fontsize=10)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of images")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_dataset_samples(dataset, classes, n_per_class=4, seed=0, save_path=None):
    """Plot ``n_per_class`` random sample images per class."""
    indices = defaultdict(list)
    for i in range(len(dataset)):
        _, label = dataset.samples[i]
        indices[label].append(i)

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(classes), n_per_class,
                             figsize=(2.2 * n_per_class, 2.4 * len(classes)))
    if len(classes) == 1:
        axes = axes[None, :]

    for r, cls in enumerate(classes):
        chosen = rng.choice(indices[cls], size=min(n_per_class, len(indices[cls])),
                            replace=False)
        for col, idx in enumerate(chosen):
            img, _ = dataset[idx]
            axes[r, col].imshow(img.permute(1, 2, 0).numpy())
            axes[r, col].axis("off")
            if col == 0:
                axes[r, col].set_ylabel(cls, fontsize=13)

    for col in range(n_per_class):
        if len(classes) > 1:
            axes[0, col].set_title(f"sample {col + 1}", fontsize=10)

    fig.suptitle("BUSI dataset samples", fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(history, title="", save_path=None):
    """Plot loss and metric curves recorded during training."""
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    def plot_pair(ax, key, label):
        ax.plot(epochs, history.get(f"train_{key}", []), marker="o", ms=3,
                label="train")
        ax.plot(epochs, history.get(f"val_{key}", []), marker="o", ms=3,
                label="val")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plot_pair(axes[0], "loss", "Loss")
    plot_pair(axes[1], "accuracy", "Accuracy")
    plot_pair(axes[2], "f1", "Macro F1")
    plot_pair(axes[3], "precision", "Precision")
    plot_pair(axes[4], "recall", "Recall")

    if history.get("lr"):
        axes[5].plot(epochs, history["lr"], marker="o", ms=3, color="green")
        axes[5].set_title("Learning Rate")
        axes[5].set_xlabel("Epoch")
        axes[5].grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm, classes, title="Confusion Matrix", normalize=True,
                          ax=None, save_path=None):
    """Plot a (optionally row-normalized) confusion matrix."""
    cm = np.asarray(cm, dtype=float)
    if normalize:
        cm = cm / cm.sum(axis=1, keepdims=True).clip(min=1e-9)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
