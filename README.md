# tiny-llm

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

From the project root, in order:

```bash
python -m tiny_llm.train_tokenizer   # data/raw/*.txt -> data/tokenizer.json
python -m tiny_llm.tokenize_dataset  # -> data/tokenized/{train,valid}.bin
python -m tiny_llm.train             # -> checkpoints/best.safetensors
python -m tiny_llm.generate          # prompt is hard-coded in generate.py
```

Settings: `src/tiny_llm/config.py`
