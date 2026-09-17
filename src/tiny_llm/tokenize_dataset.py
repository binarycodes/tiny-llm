import hashlib
import time
from collections.abc import Generator
from io import BufferedWriter
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tokenizers import Tokenizer

from tiny_llm.config import (
    DOCUMENT_SEPARATOR,
    RAW_DIR,
    SEQUENCE_SEPARATOR,
    TOKEN_DTYPE,
    TOKENIZER_FILE,
    TRAIN_FILE,
    VALID_FILE,
    VALIDATION_RATIO,
    create_directories,
)


class Split[T](NamedTuple):
    training: T
    validation: T


def iter_documents(path: Path, chunk_size: int = 4 * 1024 * 1024) -> Generator[str]:
    buffer: str = ""

    with path.open("r", encoding="utf-8") as file:
        while chunk := file.read(chunk_size):
            *documents, buffer = (buffer + chunk).split(DOCUMENT_SEPARATOR)
            yield from (doc for document in documents if (doc := document.strip()))

        if doc := buffer.strip():
            yield doc


def is_validation(data: bytes) -> bool:
    """Deterministic coin flip baked into the document text.

    Hash the text to 8 bytes, read them as an integer, scale to [0, 1) and
    compare with VALIDATION_RATIO. A good hash is uniform, so about 5% of
    documents fall below. Unlike a seeded RNG, the result depends only on the
    text. Adding or reordering files never moves a document across the split,
    and duplicate documents always land on the same side.
    """
    digest = hashlib.blake2b(data, digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64 < VALIDATION_RATIO


class Progress:
    def __init__(self, total_bytes: int, interval_seconds: float = 10.0) -> None:
        self.total_bytes = total_bytes
        self.interval_seconds = interval_seconds
        self.bytes = 0
        self.documents = 0
        self.tokens = 0
        self.start = time.perf_counter()
        self.last_report = self.start

    def update(self, byte_count: int, token_count: int) -> None:
        self.bytes += byte_count
        self.documents += 1
        self.tokens += token_count

        now = time.perf_counter()
        if now - self.last_report >= self.interval_seconds:
            self.last_report = now
            self.report()

    def report(self) -> None:
        elapsed = time.perf_counter() - self.start
        percent = min(100.0, 100.0 * self.bytes / self.total_bytes)
        rate = self.tokens / elapsed / 1e6 if elapsed else 0.0
        print(
            f"{percent:5.1f}%  {self.documents:,} documents  {self.tokens:,} tokens  "
            f"{rate:.2f}M tokens/s  {elapsed:,.0f}s",
            flush=True,
        )


def eos_token_id(tokenizer: Tokenizer) -> int:
    eos_id = tokenizer.token_to_id(SEQUENCE_SEPARATOR)
    if eos_id is None:
        raise RuntimeError(f"Tokenizer {TOKENIZER_FILE} has no {SEQUENCE_SEPARATOR} token")

    max_id = np.iinfo(TOKEN_DTYPE).max
    if tokenizer.get_vocab_size() - 1 > max_id:
        raise RuntimeError(f"Vocabulary of {tokenizer.get_vocab_size()} does not fit in {np.dtype(TOKEN_DTYPE).name}")

    return eos_id


def tokenize_file(
    tokenizer: Tokenizer,
    eos_id: int,
    file_path: Path,
    outputs: Split[BufferedWriter],
    progress: Progress,
) -> Split[int]:
    training_token_count = 0
    validation_token_count = 0

    for doc in iter_documents(file_path):
        data = doc.encode("utf-8")
        token_ids = tokenizer.encode(doc).ids
        token_ids.append(eos_id)
        tokens = np.asarray(token_ids, dtype=TOKEN_DTYPE)

        if is_validation(data):
            tokens.tofile(outputs.validation)
            validation_token_count += len(token_ids)
        else:
            tokens.tofile(outputs.training)
            training_token_count += len(token_ids)

        progress.update(len(data) + len(DOCUMENT_SEPARATOR), len(token_ids))

    return Split(training_token_count, validation_token_count)


def tokenize_raw_directory() -> Split[int]:
    files = sorted(RAW_DIR.rglob("*.txt"))
    if not files:
        raise RuntimeError(f"No .txt files found under {RAW_DIR}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    eos_id = eos_token_id(tokenizer)

    total_bytes = sum(path.stat().st_size for path in files)
    print(f"Tokenizing {len(files):,} files ({total_bytes / 1e9:.2f} GB)", flush=True)
    progress = Progress(total_bytes)

    training_token_count = 0
    validation_token_count = 0
    with open(TRAIN_FILE, "wb") as train_out, open(VALID_FILE, "wb") as valid_out:
        outputs = Split(train_out, valid_out)
        for path in files:
            counts = tokenize_file(tokenizer, eos_id, path, outputs, progress)
            training_token_count += counts.training
            validation_token_count += counts.validation
    progress.report()

    if not training_token_count or not validation_token_count:
        raise RuntimeError(
            f"Split produced {training_token_count:,} training and {validation_token_count:,} validation tokens"
        )

    return Split(training_token_count, validation_token_count)


def main() -> None:
    create_directories()

    counts = tokenize_raw_directory()

    print(f"Total tokens: {counts.training + counts.validation:,}")
    print(f"Training tokens: {counts.training:,}")
    print(f"Validation tokens: {counts.validation:,}")


if __name__ == "__main__":
    main()
