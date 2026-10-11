"""Evaluate a trained ResNet / ViT model on the BUSI dataset.

By default it loads the best checkpoint produced by ``train.py`` and reports
metrics on the test split, but any checkpoint path can be passed explicitly.

Examples:
    python evaluate.py --model resnet --depth 18
    python evaluate.py --model vit
    python evaluate.py --model resnet --checkpoint outputs/checkpoints/resnet18_best.pt
    python evaluate.py --model resnet --split val
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless evaluation

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

import config
from models import build_resnet, build_vit, build_pretrained_vit
from utils.dataset import build_loaders
from utils.metrics import AverageMeter, compute_metrics
from utils.visualization import plot_confusion_matrix


def get_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_model(args, num_classes):
    if args.model == "resnet":
        model = build_resnet(depth=args.depth, num_classes=num_classes)
        name = f"resnet{args.depth}"
    elif args.model == "vit":
        if args.pretrained:
            model = build_pretrained_vit(variant=args.vit_variant,
                                         num_classes=num_classes,
                                         image_size=args.img_size)
            name = f"vit_{args.vit_variant}"
        else:
            model = build_vit(image_size=args.img_size,
                              patch_size=args.patch_size,
                              num_classes=num_classes, dim=args.vit_dim,
                              depth=args.vit_depth, heads=args.vit_heads,
                              mlp_dim=args.vit_mlp_dim, dropout=args.vit_dropout)
            name = "vit"
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model, name


def load_checkpoint(model, ckpt_path, device):
    """Load a state_dict (or a full checkpoint dict) into the model."""
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        loss_meter.update(loss.item(), labels.size(0))
        y_true.append(labels.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(loss_meter.avg)
    return metrics, y_true, y_pred


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ResNet / ViT on BUSI")
    parser.add_argument("--model", type=str, default="resnet",
                        choices=["resnet", "vit"])
    parser.add_argument("--depth", type=int, default=config.RESNET_DEPTH,
                        help="ResNet depth (18/34/50/101)")
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="path to a checkpoint (defaults to best checkpoint)")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val"],
                        help="which split to evaluate")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction,
                        default=config.PRETRAINED,
                        help="match train.py's pretrained backbone setting")

    # ViT hyperparameters (must match training)
    parser.add_argument("--vit-variant", type=str, default=config.VIT_VARIANT,
                        choices=["b16", "b32", "l16", "l32"],
                        help="pretrained torchvision ViT variant")
    parser.add_argument("--patch-size", type=int, default=config.VIT_PATCH_SIZE)
    parser.add_argument("--vit-dim", type=int, default=config.VIT_DIM)
    parser.add_argument("--vit-depth", type=int, default=config.VIT_DEPTH)
    parser.add_argument("--vit-heads", type=int, default=config.VIT_HEADS)
    parser.add_argument("--vit-mlp-dim", type=int, default=config.VIT_MLP_DIM)
    parser.add_argument("--vit-dropout", type=float, default=config.VIT_DROPOUT)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader, classes = build_loaders(
        config.DATA_DIR, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers,
        ratios=(config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO),
        seed=args.seed)
    loader = test_loader if args.split == "test" else val_loader
    print(f"Classes: {classes}")
    print(f"Evaluating on {args.split} split: {len(loader.dataset)} samples")

    model, model_name = build_model(args, len(classes))

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"

    model = load_checkpoint(model, ckpt_path, device)
    model = model.to(device)
    print(f"Loaded checkpoint: {ckpt_path}")

    criterion = nn.CrossEntropyLoss()
    metrics, y_true, y_pred = evaluate(model, loader, criterion, device)

    print("\nOverall metrics:")
    for key, value in metrics.items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=classes,
                                digits=4, zero_division=0))

    # Save results alongside the training logs.
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_prefix = config.LOG_DIR / f"{model_name}_{args.split}"

    plot_confusion_matrix(metrics["confusion_matrix"], classes,
                          title=f"{model_name} {args.split} confusion matrix",
                          save_path=out_prefix.with_suffix(".png"))

    serializable = {k: (v.tolist() if hasattr(v, "tolist") else v)
                    for k, v in metrics.items()}
    with open(out_prefix.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nSaved confusion matrix and metrics to {config.LOG_DIR}")


if __name__ == "__main__":
    main()
