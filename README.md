# tiny-llm

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
source .venv/bin/activate
pre-commit install
```


```bash
mkdir -p ./data/raw/tinystories
curl -L -o ./data/raw/tinystories/TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
```

## Run

From the project root, in order:

```bash
train-tokenizer   # data/raw/*.txt -> data/tokenizer.json
tokenize          # -> data/tokenized/<source>.{train,valid}.bin
train             # -> checkpoints/best.safetensors
generate          # prompt is hard-coded in generate.py
```

Settings: `src/tiny_llm/config.py`

## Checks

```bash
ruff check src tools
ruff format src tools
pyright
```

The pre-commit hook runs the same checks on every commit.
