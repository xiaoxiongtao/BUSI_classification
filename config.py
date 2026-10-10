"""Central configuration for the BUSI classification project."""
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "Dataset_BUSI_with_GT"
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"

# --- Dataset ---
CLASSES = ["benign", "malignant", "normal"]
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

# --- Training ---
EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
LR_STEP_SIZE = 15
LR_GAMMA = 0.1
EARLY_STOP_PATIENCE = 15
LABEL_SMOOTHING = 0.1
PRETRAINED = True
USE_CLASS_WEIGHTS = True

# --- ResNet ---
RESNET_DEPTH = 18  # 18 / 34 / 50 / 101

# --- ViT ---
VIT_PATCH_SIZE = 16
VIT_DIM = 256
VIT_DEPTH = 6
VIT_HEADS = 8
VIT_MLP_DIM = 512
VIT_DROPOUT = 0.1
