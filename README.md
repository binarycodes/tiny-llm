# tiny-llm

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pre-commit install
```


```bash
mkdir -p ./data/raw/tinystories
curl -L -o ./data/raw/tinystories/TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
```

## Run

From the project root, in order:

```bash
uv run train-tokenizer   # data/raw/*.txt -> data/tokenizer.json
uv run tokenize          # -> data/tokenized/<source>.{train,valid}.bin; skips unchanged sources, --force redoes all
uv run train             # -> checkpoints/best.safetensors
uv run generate          # prompt is hard-coded in generate.py
```

Settings: `src/tiny_llm/config.py`

## Checks

```bash
uv run ruff check src tools
uv run ruff format src tools
uv run pyright
```

The pre-commit hook runs the same checks on every commit.
