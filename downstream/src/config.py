"""Training configuration for 3D U-Net on BraTS 2023."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

# ---------------- Dataset ----------------
JSON_PATH = os.environ.get('BRATS_JSON', os.path.join(_REPO, 'data', 'brats2023_distribution.json'))
EXPECTED_DEPTH = 80          # number of slices per case (D dimension)
INPUT_SIZE = (80, 256, 256)  # (D, H, W) — must be divisible by 16

IN_CHANNELS = 4              # T1, T1ce, T2, FLAIR
NUM_CLASSES = 3              # WT, TC, ET (multi-channel binary, sigmoid output)

MODALITY_NAMES = ['T1', 'T1ce', 'T2', 'FLAIR']
REGION_NAMES = ['WT', 'TC', 'ET']

# ---------------- Training ----------------
TRAINING_EPOCH = 100
TRAIN_BATCH_SIZE = 2     # per-GPU; total = 8 with 4 GPUs
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
WARMUP_EPOCHS = 5
NUM_WORKERS = 4

USE_AMP = True
AMP_DTYPE = 'bf16'           # 'bf16' or 'fp16'

DICE_WEIGHT = 1.0
BCE_WEIGHT = 1.0

# ---------------- Misc ----------------
SEED = 42
LOG_INTERVAL = 10            # log every N iterations
VAL_INTERVAL = 1             # run validation every N epochs
SAVE_INTERVAL = 20           # save a periodic checkpoint every N epochs

OUTPUT_DIR = os.path.join(_HERE, 'runs', 'brats_unet3d')

# ---------------- Swanlab ----------------
SWANLAB_PROJECT = '3D-UNet-BraTS'
SWANLAB_EXPERIMENT = 'unet3d_brats_4mod_80x256x256'
