"""Train a ResNet or ViT model on the BUSI dataset.

Examples:
    python train.py --model resnet --depth 18 --epochs 50
    python train.py --model vit --epochs 50
    python train.py --model resnet --depth 50 --use-class-weights
"""
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless training

import argparse
import json
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

import config
from models import build_resnet, build_vit
from utils.dataset import build_loaders
from utils.metrics import AverageMeter, compute_metrics
from utils.visualization import plot_training_curves, plot_confusion_matrix


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), labels.size(0))
        y_true.append(labels.cpu().numpy())
        y_pred.append(logits.argmax(dim=1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = float(loss_meter.avg)
    return metrics


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
    return metrics


def build_model(args, num_classes):
    if args.model == "resnet":
        model = build_resnet(depth=args.depth, num_classes=num_classes,
                             pretrained=args.pretrained)
        name = f"resnet{args.depth}"
    elif args.model == "vit":
        model = build_vit(image_size=args.img_size, patch_size=args.patch_size,
                          num_classes=num_classes, dim=args.vit_dim,
                          depth=args.vit_depth, heads=args.vit_heads,
                          mlp_dim=args.vit_mlp_dim, dropout=args.vit_dropout)
        name = "vit"
    else:
        raise ValueError(f"Unknown model: {args.model}")
    return model, name


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet / ViT on BUSI")
    parser.add_argument("--model", type=str, default="resnet",
                        choices=["resnet", "vit"])
    parser.add_argument("--depth", type=int, default=config.RESNET_DEPTH,
                        help="ResNet depth (18/34/50/101)")
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    parser.add_argument("--lr-step", type=int, default=config.LR_STEP_SIZE)
    parser.add_argument("--lr-gamma", type=float, default=config.LR_GAMMA)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-class-weights", action="store_true",
                        help="use inverse-frequency class weights in the loss")
    parser.add_argument("--label-smoothing", type=float,
                        default=config.LABEL_SMOOTHING,
                        help="label smoothing factor for the loss")
    parser.add_argument("--pretrained", action="store_true",
                        default=config.PRETRAINED,
                        help="initialize the ResNet backbone with ImageNet "
                             "weights (--no-pretrained to train from scratch)")
    parser.add_argument("--early-stop", type=int, default=config.EARLY_STOP_PATIENCE,
                        help="early-stopping patience in epochs (0 disables it)")

    # ViT hyperparameters
    parser.add_argument("--patch-size", type=int, default=config.VIT_PATCH_SIZE)
    parser.add_argument("--vit-dim", type=int, default=config.VIT_DIM)
    parser.add_argument("--vit-depth", type=int, default=config.VIT_DEPTH)
    parser.add_argument("--vit-heads", type=int, default=config.VIT_HEADS)
    parser.add_argument("--vit-mlp-dim", type=int, default=config.VIT_MLP_DIM)
    parser.add_argument("--vit-dropout", type=float, default=config.VIT_DROPOUT)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, classes = build_loaders(
        config.DATA_DIR, img_size=args.img_size, batch_size=args.batch_size,
        num_workers=args.num_workers,
        ratios=(config.TRAIN_RATIO, config.VAL_RATIO, config.TEST_RATIO),
        seed=args.seed)
    print(f"Classes: {classes}")
    print(f"Train: {len(train_loader.dataset)} | "
          f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    model, model_name = build_model(args, len(classes))
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Params: {n_params / 1e6:.2f}M")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.use_class_weights:
        train_ds = train_loader.dataset
        counter = Counter(label for _, label in train_ds.samples)
        counts = np.array([counter[c] for c in train_ds.classes], dtype=np.float32)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(counts)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).to(device))
        print("Class weights:",
              {c: round(w, 3) for c, w in zip(train_ds.classes, weights.tolist())})

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step,
                                                gamma=args.lr_gamma)

    history = {k: [] for k in [
        "train_loss", "val_loss", "train_accuracy", "val_accuracy",
        "train_precision", "val_precision", "train_recall", "val_recall",
        "train_f1", "val_f1", "lr"]}

    best_f1, best_epoch, patience = 0.0, 0, 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion,
                                        optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        for key in ["loss", "accuracy", "precision", "recall", "f1"]:
            history[f"train_{key}"].append(train_metrics[key])
            history[f"val_{key}"].append(val_metrics[key])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss {train_metrics['loss']:.4f} "
              f"acc {train_metrics['accuracy']:.4f} | "
              f"val_loss {val_metrics['loss']:.4f} "
              f"acc {val_metrics['accuracy']:.4f} "
              f"f1 {val_metrics['f1']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch, patience = val_metrics["f1"], epoch, 0
            torch.save(model.state_dict(),
                       config.CHECKPOINT_DIR / f"{model_name}_best.pt")
        else:
            patience += 1

        if args.early_stop > 0 and patience >= args.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    torch.save(model.state_dict(),
               config.CHECKPOINT_DIR / f"{model_name}_last.pt")

    hist_path = config.LOG_DIR / f"{model_name}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, title=f"{model_name} training curves",
                         save_path=config.LOG_DIR / f"{model_name}_curves.png")

    # Evaluate the best checkpoint on the test set.
    best_path = config.CHECKPOINT_DIR / f"{model_name}_best.pt"
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(f"\nBest epoch: {best_epoch} (val F1 {best_f1:.4f})")
    print("Test metrics:")
    for key, value in test_metrics.items():
        if key != "confusion_matrix":
            print(f"  {key}: {value:.4f}")

    plot_confusion_matrix(test_metrics["confusion_matrix"], classes,
                          title=f"{model_name} test confusion matrix",
                          save_path=config.LOG_DIR / f"{model_name}_confusion.png")

    print(f"\nArtifacts saved to {config.CHECKPOINT_DIR} and {config.LOG_DIR}")


if __name__ == "__main__":
    main()
