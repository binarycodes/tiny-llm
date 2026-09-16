from itertools import islice
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tiny_llm.config import (
    MINIMUM_TRAINING_FILES,
    RANDOM_SEED,
    RAW_DIR,
    TOKEN_BATCH_SIZE,
    TOKENIZER_FILE,
    TRAIN_FILE,
    VALID_FILE,
    VALIDATION_RATIO,
    create_directories,
)

create_directories()

np.random.seed(RANDOM_SEED)

tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

files = sorted(list(RAW_DIR.rglob("*.txt")))
if not files:
    raise RuntimeError(f"No .txt files found under {RAW_DIR}")
elif len(files) < MINIMUM_TRAINING_FILES:
    raise RuntimeError(
        f"Found {len(files)} .txt files under {RAW_DIR}, "
        f"need at least {MINIMUM_TRAINING_FILES}"
    )

np.random.shuffle(files)

split = int(len(files) * (1 - VALIDATION_RATIO))

train_files = files[:split]
val_files = files[split:]

if len(train_files) == 0 or len(val_files) == 0:
    raise RuntimeError(
        f"Found {len(train_files)} training files, "
        f"and {len(val_files)} validation files."
    )

eos_id = tokenizer.token_to_id("<eos>")


def batched_lines(file, batch_size: int):
    while batch := list(islice(file, batch_size)):
        yield batch


def tokenize_file(file_path, out):
    token_count = 0
    with file_path.open("r", encoding="utf-8") as file:
        for lines in batched_lines(file, TOKEN_BATCH_SIZE):
            encodings = tokenizer.encode_batch(lines)

            for encoding in encodings:
                token_ids = encoding.ids

                np.asarray(
                    token_ids,
                    dtype=np.uint32,
                ).tofile(out)

                token_count += len(token_ids)
    np.asarray(
        [eos_id],
        dtype=np.uint32,
    ).tofile(out)
    # include the eos token
    return token_count + 1


def tokenize_files(files, bin_path):
    token_count = 0
    with open(bin_path, "wb") as out:
        for path in files:
            token_count += tokenize_file(path, out)
    return token_count


training_token_count = tokenize_files(train_files, TRAIN_FILE)
validation_token_count = tokenize_files(val_files, VALID_FILE)

print(f"Total tokens: {training_token_count+validation_token_count:,}")
print(f"Training tokens: {training_token_count:,}")
print(f"Validation tokens: {validation_token_count:,}")
