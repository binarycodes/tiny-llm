from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

from tiny_llm.config import (
    RAW_DIR,
    TOKENIZER_FILE,
    VOCAB_SIZE,
    create_directories,
)

create_directories()
files = sorted(RAW_DIR.rglob("*.txt"))
if not files:
    raise RuntimeError(f"No .txt files found under {RAW_DIR}")

tokenizer = Tokenizer(
    BPE(
        unk_token="<unk>",
    )
)

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()

trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=2,
    special_tokens=[
        "<pad>",
        "<unk>",
        "<bos>",
        "<eos>",
    ],
)

print(f"Training tokenizer on {len(files):,} files...")

tokenizer.train(
    files=[str(path) for path in files],
    trainer=trainer,
)

tokenizer.save(str(TOKENIZER_FILE))

print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
print(f"Saved tokenizer to {TOKENIZER_FILE}")
