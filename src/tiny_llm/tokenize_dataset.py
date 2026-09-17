import hashlib
import time
from collections.abc import Generator
from io import BufferedWriter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from tiny_llm.config import (
    DOCUMENT_SEPARATOR,
    RAW_DIR,
    SEQUENCE_SEPARATOR,
    TOKEN_DTYPE,
    TOKENIZER_FILE,
    VALIDATION_RATIO,
    Split,
    create_directories,
    tokenized_files,
)


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
            f"{percent:5.1f}%  {self.documents:12,d} documents  {self.tokens:12,d} tokens  "
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


def tokenize_source(
    tokenizer: Tokenizer,
    eos_id: int,
    source_dir: Path,
    files: list[Path],
    progress: Progress,
) -> Split[int]:
    training_token_count = 0
    validation_token_count = 0

    paths = tokenized_files(source_dir.name)
    with open(paths.training, "wb") as train_out, open(paths.validation, "wb") as valid_out:
        outputs = Split(train_out, valid_out)
        for path in files:
            counts = tokenize_file(tokenizer, eos_id, path, outputs, progress)
            training_token_count += counts.training
            validation_token_count += counts.validation

    if not training_token_count or not validation_token_count:
        raise RuntimeError(
            f"{source_dir.name}: split produced {training_token_count:,} training "
            f"and {validation_token_count:,} validation tokens"
        )

    return Split(training_token_count, validation_token_count)


def tokenize_raw_directory() -> dict[str, Split[int]]:
    sources = {
        source_dir: sorted(source_dir.rglob("*.txt")) for source_dir in sorted(RAW_DIR.iterdir()) if source_dir.is_dir()
    }
    sources = {source_dir: files for source_dir, files in sources.items() if files}
    if not sources:
        raise RuntimeError(f"No source directories with .txt files found under {RAW_DIR}")

    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))
    eos_id = eos_token_id(tokenizer)

    file_count = sum(len(files) for files in sources.values())
    total_bytes = sum(path.stat().st_size for files in sources.values() for path in files)
    print(f"Tokenizing {file_count:,} files ({total_bytes / 1e9:.2f} GB) from {len(sources)} sources", flush=True)
    progress = Progress(total_bytes)

    counts = {
        source_dir.name: tokenize_source(tokenizer, eos_id, source_dir, files, progress)
        for source_dir, files in sources.items()
    }
    progress.report()
    return counts


def main() -> None:
    create_directories()

    counts = tokenize_raw_directory()

    for source, split in counts.items():
        print(f"{source}: training tokens {split.training:,}, validation tokens {split.validation:,}")

    training_token_count = sum(split.training for split in counts.values())
    validation_token_count = sum(split.validation for split in counts.values())
    print(f"Total tokens: {training_token_count + validation_token_count:,}")
    print(f"Training tokens: {training_token_count:,}")
    print(f"Validation tokens: {validation_token_count:,}")


if __name__ == "__main__":
    main()
