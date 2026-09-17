from pathlib import Path
from typing import NamedTuple

import numpy as np

PROJECT_ROOT = Path.cwd()

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"
TOKENIZED_DIR = DATA_DIR / "tokenized"
TOKENIZER_FILE = DATA_DIR / "tokenizer.json"


class Split[T](NamedTuple):
    training: T
    validation: T


def tokenized_files(source: str) -> Split[Path]:
    return Split(TOKENIZED_DIR / f"{source}.train.bin", TOKENIZED_DIR / f"{source}.valid.bin")


TENSORS_FILE = CHECKPOINT_DIR / "best.safetensors"
CONFIG_FILE = CHECKPOINT_DIR / "config.json"

VOCAB_SIZE = 16384
TOKEN_DTYPE = np.uint16
VALIDATION_RATIO = 0.05
RANDOM_SEED = 42

# Sampling probability per source directory under RAW_DIR, normalised at load time.
SOURCE_WEIGHTS = {
    "tinystories": 0.75,
    "technical": 0.25,
}

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
TRAIN_STEPS = 20_000
REPORT_EVERY = 50
EVAL_EVERY = 500
PATIENCE = 4


def create_directories() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
