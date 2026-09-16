import argparse
import json
from typing import cast

import mlx.core as mx
from tokenizers import Tokenizer

from tiny_llm.config import CONFIG_FILE, TENSORS_FILE, TOKENIZER_FILE
from tiny_llm.model import TinyLM


def sample(
    model: TinyLM,
    tokenizer: Tokenizer,
    context_size: int,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
):
    ids = tokenizer.encode(prompt).ids
    tokens = mx.array([ids])
    for _ in range(max_tokens):
        context = tokens[:, -context_size:]
        logits = model(context)
        logits = logits[:, -1, :] / temperature
        next_token = mx.reshape(mx.random.categorical(logits), (1, 1))
        tokens = mx.concatenate(
            [
                tokens,
                next_token,
            ],
            axis=1,
        )
        mx.eval(tokens)

    return tokenizer.decode(cast(list[int], tokens[0].tolist()))


def main():
    config = json.loads(CONFIG_FILE.read_text())
    tokenizer = Tokenizer.from_file(str(TOKENIZER_FILE))

    model = TinyLM(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        dims=config["dims"],
        num_heads=config["num_heads"],
    )

    model.load_weights(str(TENSORS_FILE))
    mx.eval(model.parameters())

    parser = argparse.ArgumentParser(description="Sample from the trained model")
    parser.add_argument("prompt", help="text to continue")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()
    print(
        sample(
            model,
            tokenizer,
            config["context_size"],
            args.prompt,
            args.max_tokens,
            args.temperature,
        )
    )


if __name__ == "__main__":
    main()
