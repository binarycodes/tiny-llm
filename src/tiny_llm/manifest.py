"""Record which inputs produced a source's tokenized files, so unchanged sources can be skipped.

A source is stale when either its raw text or the tokenizer changed: the same
text tokenized with a different vocabulary is a different dataset.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel, ValidationError

from tiny_llm.config import TOKENIZER_FILE, manifest_file, tokenized_files

CHUNK_SIZE = 4 * 1024 * 1024


def tokenizer_digest() -> str:
    return hashlib.blake2b(TOKENIZER_FILE.read_bytes()).hexdigest()


def source_digest(source_dir: Path, files: list[Path]) -> str:
    digest = hashlib.blake2b()
    for path in files:
        digest.update(str(path.relative_to(source_dir)).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            while chunk := file.read(CHUNK_SIZE):
                digest.update(chunk)
    return digest.hexdigest()


class TokenCounts(BaseModel, frozen=True):
    training: int
    validation: int


class Inputs(BaseModel, frozen=True):
    tokenizer: str
    source: str

    @classmethod
    def build(cls, source_dir: Path, files: list[Path], tokenizer: str) -> "Inputs":
        return cls(tokenizer=tokenizer, source=source_digest(source_dir, files))

    def stored_counts(self, source: str) -> TokenCounts | None:
        """Token counts from the previous run if it used the same inputs, else None."""
        paths = tokenized_files(source)
        if not (paths.training.exists() and paths.validation.exists()):
            return None
        stored = Manifest.load(source)
        if stored is None or stored.inputs != self:
            return None
        return stored.tokens

    def save(self, source: str, counts: TokenCounts) -> None:
        Manifest(inputs=self, tokens=counts).save(source)


class Manifest(BaseModel, frozen=True):
    inputs: Inputs
    tokens: TokenCounts

    @classmethod
    def load(cls, source: str) -> "Manifest | None":
        try:
            return cls.model_validate_json(manifest_file(source).read_bytes())
        except (OSError, ValidationError):
            return None

    def save(self, source: str) -> None:
        manifest_file(source).write_text(self.model_dump_json(indent=2) + "\n")
