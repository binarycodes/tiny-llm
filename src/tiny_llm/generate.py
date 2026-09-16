import argparse
import json
from pathlib import Path
import mlx.core as mx

from mlx.utils import tree_unflatten
from tokenizers import Tokenizer

from tiny_llm.config import CONFIG_FILE, TENSORS_FILE, TOKENIZER_FILE
from tiny_llm.model import TinyLM

config = json.loads(CONFIG_FILE.read_text())
tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

model = TinyLM(
    vocab_size=config["vocab_size"],
    num_layers=config["num_layers"],
    dims=config["dims"],
    num_heads=config["num_heads"],
)

weights = mx.load(TENSORS_FILE)
model.update(tree_unflatten(list(weights.items())))

mx.eval(model.parameters())


def sample(
    prompt: str,
    max_tokens=200,
    temperature=0.8,
):
    ids = tokenizer.encode(prompt).ids
    tokens = mx.array([ids])
    for _ in range(max_tokens):
        context = tokens[:, -config["context_size"] :]
        logits = model(context)
        logits = logits[:, -1, :] / temperature
        next_token = mx.random.categorical(logits)
        next_token = next_token.reshape(
            1,
            1,
        )
        tokens = mx.concatenate(
            [
                tokens,
                next_token,
            ],
            axis=1,
        )
        mx.eval(tokens)

    result = tokens[0].tolist()
    return tokenizer.decode(result)


def main():
    parser = argparse.ArgumentParser(description="Sample from the trained model")
    parser.add_argument("prompt", help="text to continue")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    print(sample(args.prompt, args.max_tokens, args.temperature))


if __name__ == "__main__":
    main()
