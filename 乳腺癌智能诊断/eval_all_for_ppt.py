"""Evaluate every checkpoint on the BUSI test split and dump a JSON summary.

Used to collect the real numbers shown in the lecture slides.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models as tv_models

import config
from models import build_resnet, build_vit, build_pretrained_vit
from utils.dataset import build_loaders
from utils.visualization import plot_confusion_matrix


def load_model(name, device):
    if name.startswith("resnet"):
        depth = int(name.replace("resnet", ""))
        m = build_resnet(depth=depth, num_classes=3, pretrained=False)
    elif name == "vit":
        m = build_vit(image_size=224, patch_size=16, num_classes=3, dim=256,
                      depth=6, heads=8, mlp_dim=512, dropout=0.1)
    elif name.startswith("vit_"):
        m = build_pretrained_vit(variant=name.split("_")[1], num_classes=3,
                                 image_size=224)
    else:
        raise ValueError(name)
    return m.to(device)


@torch.no_grad()
def run(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob, losses = [], [], [], []
    ce = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        losses.append(ce(logits, labels).item())
        prob = torch.softmax(logits, dim=1)
        y_prob.append(prob.cpu().numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
        y_true.append(labels.cpu().numpy())
    return (np.concatenate(y_true), np.concatenate(y_pred),
            np.concatenate(y_prob), float(np.mean(losses)))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    _, _, test_loader, classes = build_loaders(
        config.DATA_DIR, img_size=224, batch_size=16, num_workers=2,
        ratios=(0.7, 0.15, 0.15), seed=42)
    print("classes:", classes, "test size:", len(test_loader.dataset))

    from collections import Counter
    cnt = Counter(l for _, l in test_loader.dataset.samples)
    print("test class counts:", dict(cnt))

    names = ["resnet18", "resnet34", "resnet50", "resnet101", "vit_b16", "vit"]
    out = {}
    for name in names:
        ckpt = config.CHECKPOINT_DIR / f"{name}_best.pt"
        if not ckpt.exists():
            print("missing", ckpt)
            continue
        model = load_model(name, device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        n_params = sum(p.numel() for p in model.parameters())
        y_true, y_pred, y_prob, loss = run(model, test_loader, device)

        K = len(classes)
        cm = np.zeros((K, K), dtype=int)
        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        acc = float((y_true == y_pred).mean())
        per_class = {}
        for i, c in enumerate(classes):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            tn = cm.sum() - tp - fp - fn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            per_class[c] = {"support": int(cm[i, :].sum()), "precision": prec,
                            "recall": rec, "f1": f1, "specificity":
                            tn / (tn + fp) if tn + fp else 0.0}
        macro = {m: float(np.mean([per_class[c][m] for c in classes]))
                 for m in ["precision", "recall", "f1"]}

        out[name] = {
            "params_M": round(n_params / 1e6, 2),
            "test_loss": round(loss, 4),
            "accuracy": round(acc, 4),
            "macro_precision": round(macro["precision"], 4),
            "macro_recall": round(macro["recall"], 4),
            "macro_f1": round(macro["f1"], 4),
            "per_class": {c: {k: (round(v, 4) if isinstance(v, float) else v)
                              for k, v in d.items()}
                          for c, d in per_class.items()},
            "confusion_matrix": cm.tolist(),
        }
        plot_confusion_matrix(cm, classes, title=f"{name} test confusion matrix",
                              save_path=config.LOG_DIR / f"{name}_confusion.png")
        print(f"{name:12s} acc={acc:.4f} f1={macro['f1']:.4f} "
              f"params={n_params/1e6:.1f}M")

    (config.LOG_DIR / "ppt_model_summary.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved ->", config.LOG_DIR / "ppt_model_summary.json")


if __name__ == "__main__":
    main()
