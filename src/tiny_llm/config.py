from pathlib import Path

import numpy as np

PROJECT_ROOT = Path.cwd()

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
TOKENIZED_DIR = DATA_DIR / "tokenized"
TOKENIZER_FILE = DATA_DIR / "tokenizer.json"

TRAIN_FILE = TOKENIZED_DIR / "train.bin"
VALID_FILE = TOKENIZED_DIR / "valid.bin"

TENSORS_FILE = CHECKPOINT_DIR / "best.safetensors"
CONFIG_FILE = CHECKPOINT_DIR / "config.json"

VOCAB_SIZE = 16384
TOKEN_DTYPE = np.uint16
VALIDATION_RATIO = 0.05
RANDOM_SEED = 42

SEQUENCE_SEPARATOR = "<eos>"
DOCUMENT_SEPARATOR = "<|endoftext|>"

SPECIAL_TOKENS = [
    "<pad>",
    "<unk>",
    "<bos>",
    SEQUENCE_SEPARATOR,
]

CONTEXT_SIZE = 256
BATCH_SIZE = 16
NUM_LAYERS = 8
DIMS = 384
NUM_HEADS = 6
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1
TRAIN_STEPS = 10_000
REPORT_EVERY = 50
EVAL_EVERY = 500
PATIENCE = 4


def create_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
