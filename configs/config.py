"""
Central configuration for the congestion prediction project.
Edit paths and hyperparameters here.
"""
import os

# ============================================================
# DATA PATHS — Update these to match your setup
# ============================================================
# Path relative to the project root (congestion_prediction/)
FEATURE_DIR = "data/circuitnet/training_set/congestion/feature"
LABEL_DIR = "data/circuitnet/training_set/congestion/label"

# Feature channel names (for plotting and analysis)
FEATURE_NAMES = ["macro_region", "RUDY", "RUDY_pin"]

# ============================================================
# DATA PROPERTIES (from generate_training_set.py output)
# ============================================================
NUM_INPUT_CHANNELS = 3    # macro_region, RUDY, RUDY_pin
NUM_OUTPUT_CHANNELS = 1   # combined H+V congestion overflow
IMAGE_SIZE = 256

# ============================================================
# DATA SPLIT
# ============================================================
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SCHEDULER_PATIENCE = 5    # Reduce LR after this many epochs without improvement
EARLY_STOP_PATIENCE = 10  # Stop training after this many epochs without improvement

# ============================================================
# OUTPUT
# ============================================================
RESULTS_DIR = "./results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")
