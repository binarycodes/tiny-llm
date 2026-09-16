from io import BufferedWriter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tiny_llm.config import (
    MINIMUM_TRAINING_FILES,
    RANDOM_SEED,
    RAW_DIR,
    TOKENIZER_FILE,
    TRAIN_FILE,
    VALID_FILE,
    VALIDATION_RATIO,
    create_directories,
)


def tokenize_file(tokenizer: Tokenizer, eos_id: int, file_path: Path, out: BufferedWriter):
    with file_path.open("r", encoding="utf-8") as file:
        file_text = file.read()

    token_ids = tokenizer.encode(file_text).ids
    token_ids.append(eos_id)

    np.asarray(
        token_ids,
        dtype=np.uint32,
    ).tofile(out)

    return len(token_ids)


def tokenize_files(tokenizer: Tokenizer, files: list[Path], bin_path: Path):
    eos_id = tokenizer.token_to_id("<eos>")
    if eos_id is None:
        raise RuntimeError(f"Tokenizer {TOKENIZER_FILE} has no <eos> token")

    token_count = 0
    with open(bin_path, "wb") as out:
        for path in files:
            token_count += tokenize_file(tokenizer, eos_id, path, out)
    return token_count


def split_source(rng: np.random.Generator, source_dir: Path, validation_ratio: float):
    files = sorted(source_dir.rglob("*.txt"))

    if not files:
        raise RuntimeError(f"No .txt files found under {source_dir}")
    elif len(files) < MINIMUM_TRAINING_FILES:
        raise RuntimeError(f"Found {len(files)} .txt files under {source_dir}, need at least {MINIMUM_TRAINING_FILES}")

    rng.shuffle(files)

    split = int(len(files) * (1.0 - validation_ratio))
    split = max(1, min(split, len(files) - 1))

    training_files = files[:split]
    validation_files = files[split:]

    if len(training_files) == 0 or len(validation_files) == 0:
        raise RuntimeError(f"Found {len(training_files)} training files, and {len(validation_files)} validation files.")

    return training_files, validation_files


def split_raw_to_source(rng: np.random.Generator) -> tuple[list[Path], list[Path]]:
    training_files: list[Path] = []
    validation_files: list[Path] = []

    for source_dir in sorted(RAW_DIR.iterdir()):
        if not source_dir.is_dir():
            continue
        source_training, source_validation = split_source(rng, source_dir, VALIDATION_RATIO)
        training_files.extend(source_training)
        validation_files.extend(source_validation)

    return training_files, validation_files


def main() -> None:
    create_directories()

    rng = np.random.default_rng(RANDOM_SEED)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    training_files, validation_files = split_raw_to_source(rng)
    training_token_count = tokenize_files(tokenizer, training_files, TRAIN_FILE)
    validation_token_count = tokenize_files(tokenizer, validation_files, VALID_FILE)

    print(f"Total tokens: {training_token_count + validation_token_count:,}")
    print(f"Training tokens: {training_token_count:,}")
    print(f"Validation tokens: {validation_token_count:,}")


if __name__ == "__main__":
    main()
